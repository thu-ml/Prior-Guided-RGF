CUDA_NUM=$1
OUTPUT_DIR=$2
METHOD=$3
NORM=$4
MODEL=$5

CUDA_VISIBLE_DEVICES="${CUDA_NUM}" nohup python attack.py \
  --model="${MODEL}" \
  --input_dir=images \
  --output_dir="${OUTPUT_DIR}" \
  --norm="${NORM}" --method="${METHOD}" --show_true --show_loss > "${OUTPUT_DIR}"/nohup.out &
