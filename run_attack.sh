
OUTPUT_DIR=$1

python attack.py \
  --model=inception-v3 \
  --input_dir=images \
  --output_dir="${OUTPUT_DIR}" \
  --norm=l2 --method=biased --show_loss
