
INPUT_DIR=$1
OUTPUT_DIR=$2

python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path=inception_v3.ckpt \
  --norm=l2 --method=biased --show_loss
