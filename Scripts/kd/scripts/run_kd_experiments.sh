# KD Experiment Runner
# Usage: bash run_kd_experiment.sh <alpha>  e.g.  bash run_kd_experiment.sh 0.5
#
# This script:
#   1. Generates soft labels for the given alpha
#   2. Creates a Darknet .data config pointing to KD labels
#   3. Starts student retraining with KD labels
#
# Prerequisites:
#   - Darknet compiled and in PATH (or set DARKNET_BIN below)
#   - Teacher weights at TEACHER_WEIGHTS path
#   - Dataset at DATASET_DIR

set -e

ALPHA=${1:-0.5}
ALPHA_TAG=$(echo "$ALPHA" | tr -d '.')   # 0.5 -> 05

# ── Paths (adjust to your environment) ───────────────────────────────────────
DARKNET_BIN="./darknet"                      # path to darknet executable
TEACHER_CFG="mycfg/yolov4-tiny_helmet.cfg"
TEACHER_WEIGHTS="mybackup/yolov4-tiny_helmet_final.weights"
STUDENT_CFG="mycfg/yolofv1_helmetv2_reduce_filter_gray_sam_ufv2.cfg"   # best student cfg
TRAIN_LIST="helmet_datasetv2_1/train.txt"
GT_LABEL_DIR="helmet_datasetv2_1/labels"
DATASET_DIR="helmet_datasetv2_1"
OUTPUT_LABEL_DIR="helmet_kd_labels_a${ALPHA_TAG}"
KD_DATA_CFG="helmet_kd_a${ALPHA_TAG}.data"
BACKUP_DIR="backup_kd_a${ALPHA_TAG}"

echo "============================================="
echo " KD Experiment  alpha=$ALPHA"
echo "============================================="

# ── Step 1: Generate soft labels ─────────────────────────────────────────────
echo "[1/3] Generating soft labels (alpha=$ALPHA)..."
python Scripts/generate_soft_labels.py \
    --teacher_cfg     "$TEACHER_CFG" \
    --teacher_weights "$TEACHER_WEIGHTS" \
    --train_list      "$TRAIN_LIST" \
    --gt_label_dir    "$GT_LABEL_DIR" \
    --output_dir      "$OUTPUT_LABEL_DIR" \
    --alpha           "$ALPHA" \
    --conf_thresh     0.30 \
    --nms_thresh      0.45

# ── Step 2: Create .data config ──────────────────────────────────────────────
echo "[2/3] Creating Darknet .data config: $KD_DATA_CFG"
CLASSES=$(grep "^classes" "$DATASET_DIR/helmetv2.data" | head -1 | awk -F= '{print $2}' | tr -d ' ')
VALID_LIST=$(grep "^valid" "$DATASET_DIR/helmetv2.data" | head -1 | awk -F= '{print $2}' | tr -d ' ')
NAMES_FILE=$(grep "^names" "$DATASET_DIR/helmetv2.data" | head -1 | awk -F= '{print $2}' | tr -d ' ')

# Create a modified train list pointing to KD label dir
KD_TRAIN_LIST="helmet_kd_a${ALPHA_TAG}_train.txt"
cp "$TRAIN_LIST" "$KD_TRAIN_LIST"

mkdir -p "$BACKUP_DIR"

cat > "$KD_DATA_CFG" <<EOF
classes = $CLASSES
train   = $KD_TRAIN_LIST
valid   = $VALID_LIST
names   = $NAMES_FILE
backup  = $BACKUP_DIR
labels  = $OUTPUT_LABEL_DIR
EOF

echo "  Created: $KD_DATA_CFG"
echo "  Backup dir: $BACKUP_DIR"

# ── Step 3: Retrain student ───────────────────────────────────────────────────
echo "[3/3] Starting student retraining with KD labels..."
echo "  Student cfg : $STUDENT_CFG"
echo "  KD data     : $KD_DATA_CFG"
echo "  Starting from best baseline weights..."

BASELINE_WEIGHTS="mybackup/yolofv1_helmetv2_reduce_filter_gray_sam_ufv2_best.weights"

$DARKNET_BIN detector train \
    "$KD_DATA_CFG" \
    "$STUDENT_CFG" \
    "$BASELINE_WEIGHTS" \
    -dont_show \
    -map \
    2>&1 | tee "kd_train_a${ALPHA_TAG}.log"

echo "============================================="
echo " KD Training complete for alpha=$ALPHA"
echo " Weights saved to: $BACKUP_DIR"
echo " Log: kd_train_a${ALPHA_TAG}.log"
echo "============================================="

