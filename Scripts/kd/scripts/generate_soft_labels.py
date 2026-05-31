"""
generate_soft_labels.py
-----------------------
Knowledge Distillation via Teacher-Supervised Augmentation

Strategy:
    The dataset is fully annotated -- teacher cannot add new boxes to original images
    (it predicts same locations as GT). Instead, we generate AUGMENTED versions of
    training images (flip, crop, brightness) that have no GT labels, then run the
    TEACHER on them to produce pseudo-labels. The student trains on both:
        - original images with GT labels       (hard supervision)
        - augmented images with teacher labels (soft/KD supervision)

    This is Response-Based KD via data augmentation -- publishable, practical,
    no Darknet modification required.

Teacher : yolov4-tiny_helmet.cfg  (RGB, 320x320, high accuracy)
Student : yolofv1_helmetv2_reduce_filter_gray_sam  (Grayscale, 320x320, MCU-deployable)
Classes : 0=head, 1=helmet

Label format modes (--label_mode):
    hard      : Standard 5-field YOLO labels  "cls cx cy bw bh"
                All boxes treated equally regardless of teacher confidence.
                Use with standard unmodified Darknet.

    soft      : 6-field labels  "cls cx cy bw bh conf"
                Teacher confidence saved as 6th field.
                Standard Darknet ignores the 6th field (safe to use).
                A soft-weight-aware Darknet can use conf to weight the loss,
                implementing proper soft-label KD (DKD/DIST style).

    filtered  : Like 'hard' but raises the acceptance threshold to --hard_thresh
                (default 0.50). Removes noisy low-confidence pseudo-labels.
                Best immediate improvement without any Darknet modification.

Recommended strategy:
    1st run: --label_mode filtered --hard_thresh 0.50 --conf_thresh 0.35
             Keeps only high-confidence teacher boxes -> cleaner pseudo-labels
    2nd run: --label_mode soft --conf_thresh 0.35
             Saves soft weights for future soft-label-aware training

Output:
    kd_aug_images/   - augmented image files (.jpg)
    kd_aug_labels/   - teacher pseudo-labels for augmented images (.txt)
    kd_aug_train.txt - combined train list (original GT + augmented teacher-labeled)

Usage:
    python generate_soft_labels.py \
        --teacher_cfg     mycfg/yolov4-tiny_helmet.cfg \
        --teacher_weights mybackup/yolov4-tiny_helmet_final.weights \
        --names_file      helmet_datasetv2_1/helmetv2.names \
        --train_list      helmet_datasetv2_1/train.txt \
        --output_img_dir  kd_aug_images \
        --output_lbl_dir  kd_aug_labels \
        --output_list     kd_aug_train.txt \
        --conf_thresh     0.35 \
        --augs_per_image  2 \
        --label_mode      filtered \
        --hard_thresh     0.50

Experiment plan (augs_per_image controls how much extra data):
    augs_per_image=1  ->  +4912  augmented images  (1x extra)   conservative
    augs_per_image=2  ->  +9824  augmented images  (2x extra)   recommended
    augs_per_image=3  ->  +14736 augmented images  (3x extra)   aggressive
"""

import darknet
import cv2
import os
import argparse
import random
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Augmentation functions
# Each returns (augmented_image_BGR, transform_params)
# transform_params is used to adjust GT boxes if needed (not used here since
# teacher re-labels the augmented image directly)
# ---------------------------------------------------------------------------

def aug_horizontal_flip(img):
    return cv2.flip(img, 1), "hflip"


def aug_brightness(img):
    factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), f"bright_{factor:.2f}"


def aug_random_crop(img, min_ratio=0.65, max_ratio=0.95):
    h, w = img.shape[:2]
    ratio = random.uniform(min_ratio, max_ratio)
    new_h, new_w = int(h * ratio), int(w * ratio)
    top  = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = img[top:top+new_h, left:left+new_w]
    # Resize back to original size so teacher input is consistent
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized, f"crop_{ratio:.2f}"


def aug_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy, "noise"


def aug_hue_shift(img):
    shift = random.randint(-15, 15)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), f"hue_{shift}"


# Pool of augmentations to sample from
AUG_POOL = [
    aug_horizontal_flip,
    aug_brightness,
    aug_random_crop,
    aug_noise,
    aug_hue_shift,
]


def apply_random_aug(img):
    """Apply 1-2 random augmentations from the pool."""
    n = random.randint(1, 2)
    chosen = random.sample(AUG_POOL, n)
    tags = []
    for fn in chosen:
        img, tag = fn(img)
        tags.append(tag)
    return img, "_".join(tags)


# ---------------------------------------------------------------------------
# Teacher inference
# ---------------------------------------------------------------------------

def load_teacher(cfg, weights):
    net = darknet.load_net_custom(cfg.encode(), weights.encode(), 0, 1)
    w = darknet.network_width(net)
    h = darknet.network_height(net)
    return net, w, h


def load_class_names(names_file):
    with open(names_file, "r") as f:
        return [l.strip() for l in f if l.strip()]


def teacher_predict(net, net_w, net_h, img_bgr, class_names, conf_thresh, nms_thresh=0.45):
    """
    Run teacher on a BGR numpy image.
    Returns list of (class_id, confidence, cx, cy, bw, bh) normalized to [0,1].
    class_names must be passed so darknet.detect_image returns proper labels.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (net_w, net_h))

    dk_img = darknet.make_image(net_w, net_h, 3)
    darknet.copy_image_from_bytes(dk_img, img_resized.tobytes())
    detections = darknet.detect_image(net, class_names, dk_img,
                                      thresh=conf_thresh,
                                      hier_thresh=0.5,
                                      nms=nms_thresh)
    darknet.free_image(dk_img)

    # Build name->id map from the loaded class names
    name_to_id = {name.lower(): i for i, name in enumerate(class_names)}

    results = []
    for label, conf, (x, y, w, h) in detections:
        label_str = str(label).strip().lower()
        if label_str in name_to_id:
            class_id = name_to_id[label_str]
        else:
            try:
                class_id = int(label_str)
            except ValueError:
                continue  # unknown label, skip

        cx = max(0.0, min(1.0, x / net_w))
        cy = max(0.0, min(1.0, y / net_h))
        bw = max(0.001, min(1.0, w / net_w))
        bh = max(0.001, min(1.0, h / net_h))
        results.append((class_id, float(conf), cx, cy, bw, bh))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KD via Teacher-Supervised Augmentation")
    parser.add_argument("--teacher_cfg",      required=True)
    parser.add_argument("--teacher_weights",  required=True)
    parser.add_argument("--names_file",       required=True,
                        help="Path to .names file (e.g. helmet_datasetv2_1/helmetv2.names)")
    parser.add_argument("--train_list",       required=True,
                        help="Original train.txt (one image path per line)")
    parser.add_argument("--output_img_dir",   default="kd_aug_images")
    parser.add_argument("--output_lbl_dir",   default="kd_aug_labels")
    parser.add_argument("--output_list",      default="kd_aug_train.txt",
                        help="Output combined train list (original + augmented)")
    parser.add_argument("--conf_thresh",      type=float, default=0.35,
                        help="Minimum teacher confidence to detect a box (pre-NMS gate)")
    parser.add_argument("--nms_thresh",       type=float, default=0.45)
    parser.add_argument("--augs_per_image",   type=int, default=2,
                        help="How many augmented versions to generate per image")
    parser.add_argument("--min_boxes",        type=int, default=1,
                        help="Discard augmented image if teacher finds fewer than this many boxes")
    parser.add_argument("--label_mode",       default="filtered",
                        choices=["hard", "soft", "filtered"],
                        help=(
                            "hard: standard 5-field YOLO labels (original behavior). "
                            "soft: 6-field labels with confidence appended (recommended for soft-label KD). "
                            "filtered: 5-field labels but only keeps boxes above --hard_thresh "
                            "(best immediate improvement without Darknet modification)."
                        ))
    parser.add_argument("--hard_thresh",      type=float, default=0.50,
                        help="[filtered mode only] Minimum confidence to save a label box. "
                             "Higher = cleaner pseudo-labels. Typical range: 0.45-0.70.")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_lbl_dir, exist_ok=True)

    print(f"[KD-AUG] Loading teacher: {args.teacher_cfg}")
    teacher_net, net_w, net_h = load_teacher(args.teacher_cfg, args.teacher_weights)
    class_names = load_class_names(args.names_file)
    print(f"[KD-AUG] Teacher: {net_w}x{net_h} RGB  classes={class_names}")
    print(f"[KD-AUG] augs_per_image={args.augs_per_image}, conf_thresh={args.conf_thresh}")
    print(f"[KD-AUG] label_mode={args.label_mode}", end="")
    if args.label_mode == "filtered":
        print(f"  hard_thresh={args.hard_thresh}")
    elif args.label_mode == "soft":
        print("  (6-field: cls cx cy bw bh conf)")
    else:
        print("  (5-field standard YOLO)")

    with open(args.train_list) as f:
        original_paths = [l.strip() for l in f if l.strip()]

    total = len(original_paths)
    print(f"[KD-AUG] Original training images: {total}")
    print(f"[KD-AUG] Expected augmented images: ~{total * args.augs_per_image}")

    aug_paths = []  # paths of accepted augmented images
    stats = {
        "generated": 0, "accepted": 0,
        "rejected_no_boxes": 0, "errors": 0,
        "boxes_total": 0, "boxes_filtered": 0,
        "conf_sum": 0.0,
    }

    for idx, img_path in enumerate(original_paths):
        if (idx + 1) % 200 == 0 or idx == 0:
            print(f"[KD-AUG] {idx+1}/{total}  accepted={stats['accepted']}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            stats["errors"] += 1
            continue

        for aug_i in range(args.augs_per_image):
            stats["generated"] += 1

            # 1. Augment
            try:
                aug_img, aug_tag = apply_random_aug(img_bgr.copy())
            except Exception as e:
                stats["errors"] += 1
                continue

            # 2. Teacher labels augmented image
            try:
                preds = teacher_predict(teacher_net, net_w, net_h,
                                        aug_img, class_names, args.conf_thresh, args.nms_thresh)
            except Exception as e:
                stats["errors"] += 1
                continue

            # 3. Reject if too few boxes (likely background crop or bad aug)
            if len(preds) < args.min_boxes:
                stats["rejected_no_boxes"] += 1
                continue

            # 4. Filter by label mode before deciding to save
            if args.label_mode == "filtered":
                save_preds = [(cls, conf, cx, cy, bw, bh)
                              for (cls, conf, cx, cy, bw, bh) in preds
                              if conf >= args.hard_thresh]
                stats["boxes_filtered"] += len(preds) - len(save_preds)
            else:
                save_preds = preds

            # Still require min_boxes after filtering
            if len(save_preds) < args.min_boxes:
                stats["rejected_no_boxes"] += 1
                continue

            stats["boxes_total"] += len(save_preds)
            stats["conf_sum"] += sum(c for (_, c, *_) in save_preds)

            # 5. Save augmented image
            stem = Path(img_path).stem
            out_name = f"{stem}_kd_{aug_i}_{aug_tag}"
            img_out = os.path.join(args.output_img_dir, out_name + ".jpg")
            cv2.imwrite(img_out, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 6. Save pseudo-label
            #    hard/filtered : "cls cx cy bw bh"          (5 fields, standard YOLO)
            #    soft          : "cls cx cy bw bh conf"      (6 fields; standard Darknet
            #                                                  ignores the 6th field safely;
            #                                                  a soft-weight-aware Darknet
            #                                                  uses conf to scale the loss)
            lbl_out = os.path.join(args.output_lbl_dir, out_name + ".txt")
            with open(lbl_out, "w") as f:
                for (cls, conf, cx, cy, bw, bh) in save_preds:
                    if args.label_mode == "soft":
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
                    else:
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            aug_paths.append(img_out)
            stats["accepted"] += 1

    # 7. Write combined train list: original + augmented
    with open(args.output_list, "w") as f:
        for p in original_paths:
            f.write(p + "\n")
        for p in aug_paths:
            f.write(p + "\n")

    accept_rate = stats["accepted"] / max(stats["generated"], 1) * 100
    avg_conf = stats["conf_sum"] / max(stats["boxes_total"], 1)
    print("\n[KD-AUG] ── Summary ────────────────────────────────────")
    print(f"  Original images     : {total}")
    print(f"  Augmented generated : {stats['generated']}")
    print(f"  Accepted (has boxes): {stats['accepted']}  ({accept_rate:.1f}%)")
    print(f"  Rejected (no boxes) : {stats['rejected_no_boxes']}")
    print(f"  Errors              : {stats['errors']}")
    print(f"  Total boxes saved   : {stats['boxes_total']}")
    if args.label_mode == "filtered":
        print(f"  Boxes filtered (<{args.hard_thresh:.2f} conf): {stats['boxes_filtered']}")
    print(f"  Avg confidence      : {avg_conf:.3f}")
    print(f"  Label mode          : {args.label_mode}")
    print(f"  Total in train list : {total + stats['accepted']}")
    print(f"  Output train list   : {args.output_list}")
    print(f"  Augmented images    : {args.output_img_dir}/")
    print(f"  Pseudo-labels       : {args.output_lbl_dir}/")
    print("[KD-AUG] ─────────────────────────────────────────────────")
    print("[KD-AUG] Done.")
    if args.label_mode == "soft":
        print("[KD-AUG] NOTE: 6-field labels saved. Standard Darknet ignores the 6th field.")
        print("[KD-AUG]       To use soft weights, modify region_layer.c:")
        print("[KD-AUG]         delta_region_box()  -- multiply loss by conf weight")
        print("[KD-AUG]         delta_region_class() -- multiply loss by conf weight")
    print(f"[KD-AUG] Next: create a Darknet .data config with train={args.output_list}")
    print(f"[KD-AUG]       and labels pointing to both GT and {args.output_lbl_dir}/")


if __name__ == "__main__":
    main()
