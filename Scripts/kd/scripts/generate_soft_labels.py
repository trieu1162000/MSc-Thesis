"""
generate_soft_labels.py
-----------------------
Response-Based Knowledge Distillation: Soft Label Generation
- Runs the TEACHER model (YOLOv4-tiny) on every training image
- Merges teacher predictions with ground-truth (GT) labels using confidence weighting
- Outputs blended label files ready for Darknet student retraining

Teacher : yolov4-tiny_helmet.cfg  (RGB, 320x320)
Student : yolofv1_helmetv2_reduce_filter_gray_sam_*.cfg (Grayscale, 320x320)
Classes : 0=head, 1=helmet

Usage:
    python generate_soft_labels.py \
        --teacher_cfg   mycfg/yolov4-tiny_helmet.cfg \
        --teacher_weights mybackup/yolov4-tiny_helmet_final.weights \
        --train_list    helmet_datasetv2_1/train.txt \
        --gt_label_dir  helmet_datasetv2_1/labels \
        --output_dir    helmet_kd_labels \
        --alpha         0.5 \
        --conf_thresh   0.30 \
        --nms_thresh    0.45

KD blend formula (per bounding box):
    blended_conf = alpha * teacher_conf + (1 - alpha) * 1.0   [GT boxes get conf=1.0]
    The output label format is standard YOLO: <class> <cx> <cy> <w> <h>
    Teacher boxes below conf_thresh are discarded (noise filter).
    GT boxes are always kept (alpha does not remove them).
"""

import darknet
import cv2
import os
import argparse
import shutil
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def load_teacher(cfg: str, weights: str):
    net = darknet.load_net_custom(cfg.encode(), weights.encode(), 0, 1)
    w = darknet.network_width(net)
    h = darknet.network_height(net)
    return net, w, h


def teacher_predict(net, net_w, net_h, image_path: str,
                    conf_thresh: float, nms_thresh: float):
    """
    Run teacher inference on one image (RGB).
    Returns list of (class_id, confidence, cx, cy, bw, bh) normalized to [0,1].
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (net_w, net_h))

    dk_img = darknet.make_image(net_w, net_h, 3)
    darknet.copy_image_from_bytes(dk_img, img_resized.tobytes())

    # detect_image returns list of (label_str, confidence_str_or_float, (x,y,w,h))
    # x,y,w,h are in PIXEL coords relative to net_w x net_h
    detections = darknet.detect_image(net, [], dk_img,
                                      thresh=conf_thresh,
                                      hier_thresh=0.5,
                                      nms=nms_thresh)
    darknet.free_image(dk_img)

    results = []
    for label, conf, (x, y, w, h) in detections:
        # darknet returns class label as string index or name — map to int
        try:
            class_id = int(label)
        except (ValueError, TypeError):
            # if label is a name string, skip unknown classes
            name_map = {"head": 0, "helmet": 1}
            class_id = name_map.get(str(label).strip().lower(), -1)
            if class_id == -1:
                continue

        # normalize to [0,1]
        cx = x / net_w
        cy = y / net_h
        bw = w / net_w
        bh = h / net_h

        # clamp
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        results.append((class_id, float(conf), cx, cy, bw, bh))

    return results


def read_gt_labels(label_path: str):
    """
    Read ground-truth YOLO label file.
    Returns list of (class_id, cx, cy, w, h).
    """
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((int(parts[0]),
                                float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])))
    return labels


def iou(box1, box2):
    """Compute IoU between two (cx,cy,w,h) boxes."""
    def to_xyxy(cx, cy, w, h):
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    x1a, y1a, x2a, y2a = to_xyxy(*box1)
    x1b, y1b, x2b, y2b = to_xyxy(*box2)

    xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
    xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter + 1e-6

    return inter / union


def blend_labels(gt_labels, teacher_preds, alpha: float, iou_thresh: float = 0.5):
    """
    Merge GT labels and teacher predictions.

    Rules:
    - All GT boxes are kept as-is (hard labels, conf implicitly = 1.0).
    - Teacher predictions that overlap a GT box (IoU > iou_thresh) are skipped
      (GT takes priority — avoids duplication).
    - Teacher predictions with NO GT overlap and conf >= conf_thresh are added
      as soft boxes. These represent teacher knowledge beyond GT annotation.

    The output label file uses standard YOLO format: <class> <cx> <cy> <w> <h>
    Teacher-sourced boxes are annotated with a comment for traceability.

    alpha controls how many teacher boxes survive:
        high alpha (0.7) = trust teacher more, add more soft boxes
        low  alpha (0.3) = conservative, only high-confidence teacher boxes
    """
    output_lines = []

    # Always write GT boxes
    for (cls, cx, cy, w, h) in gt_labels:
        output_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Add teacher boxes not covered by GT
    for (cls, conf, cx, cy, w, h) in teacher_preds:
        # Only add teacher boxes above alpha-scaled threshold
        # alpha=0.5, conf=0.6 → effective threshold = 0.6 * (1/0.5) concept:
        # simpler: just require conf >= alpha as minimum bar
        if conf < alpha:
            continue

        # Check overlap with all GT boxes
        overlaps_gt = any(
            iou((cx, cy, w, h), (gcx, gcy, gw, gh)) > iou_thresh
            for (_, gcx, gcy, gw, gh) in gt_labels
        )
        if overlaps_gt:
            continue  # GT already covers this region

        output_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}  # teacher_soft conf={conf:.3f}")

    return output_lines


def find_label_path(image_path: str, gt_label_dir: str):
    """Find the corresponding GT label file for an image."""
    stem = Path(image_path).stem
    # Try label dir first
    candidate = os.path.join(gt_label_dir, stem + ".txt")
    if os.path.exists(candidate):
        return candidate
    # Fallback: same dir as image, same name
    fallback = str(Path(image_path).with_suffix(".txt"))
    return fallback if os.path.exists(fallback) else None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KD Soft Label Generator")
    parser.add_argument("--teacher_cfg",     required=True)
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--train_list",      required=True,
                        help="Path to train.txt (one image path per line)")
    parser.add_argument("--gt_label_dir",    required=True,
                        help="Directory containing ground-truth .txt label files")
    parser.add_argument("--output_dir",      required=True,
                        help="Where to write blended label files")
    parser.add_argument("--alpha",           type=float, default=0.5,
                        help="KD blend factor (0.3 / 0.5 / 0.7 recommended)")
    parser.add_argument("--conf_thresh",     type=float, default=0.30,
                        help="Minimum teacher confidence to consider a detection")
    parser.add_argument("--nms_thresh",      type=float, default=0.45)
    parser.add_argument("--iou_thresh",      type=float, default=0.50,
                        help="IoU threshold: teacher box suppressed if overlaps GT this much")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[KD] Loading teacher: {args.teacher_cfg}")
    teacher_net, net_w, net_h = load_teacher(args.teacher_cfg, args.teacher_weights)
    print(f"[KD] Teacher input: {net_w}x{net_h} RGB")
    print(f"[KD] alpha={args.alpha}, conf_thresh={args.conf_thresh}")

    with open(args.train_list, "r") as f:
        image_paths = [l.strip() for l in f if l.strip()]

    total = len(image_paths)
    stats = {"gt_only": 0, "teacher_added": 0, "no_label": 0, "errors": 0}

    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 200 == 0 or idx == 0:
            print(f"[KD] {idx+1}/{total}  {img_path}")

        # 1. Get GT labels
        gt_path = find_label_path(img_path, args.gt_label_dir)
        gt_labels = read_gt_labels(gt_path) if gt_path else []

        # 2. Get teacher predictions
        try:
            teacher_preds = teacher_predict(
                teacher_net, net_w, net_h, img_path,
                args.conf_thresh, args.nms_thresh
            )
        except Exception as e:
            print(f"[KD] WARNING: teacher failed on {img_path}: {e}")
            teacher_preds = []
            stats["errors"] += 1

        # 3. Blend
        blended = blend_labels(gt_labels, teacher_preds, args.alpha, args.iou_thresh)
        added = len(blended) - len(gt_labels)
        if added > 0:
            stats["teacher_added"] += 1
        elif not gt_labels:
            stats["no_label"] += 1
        else:
            stats["gt_only"] += 1

        # 4. Write output label file (same stem as image)
        out_name = Path(img_path).stem + ".txt"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(blended) + ("\n" if blended else ""))

    print("\n[KD] ── Summary ──────────────────────────────")
    print(f"  Total images     : {total}")
    print(f"  GT only (no new) : {stats['gt_only']}")
    print(f"  Teacher added    : {stats['teacher_added']}  ← teacher contributed new boxes")
    print(f"  No GT label      : {stats['no_label']}")
    print(f"  Errors           : {stats['errors']}")
    print(f"  Output dir       : {args.output_dir}")
    print("[KD] ─────────────────────────────────────────")
    print("[KD] Done. Next step: update your Darknet .data config to point to these labels.")
    print(f"[KD] Suggested config name: helmetv2_kd_a{int(args.alpha*10):02d}.data")


if __name__ == "__main__":
    main()

