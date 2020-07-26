CUDA_NUM=$1
OUTPUT_DIR=$2
METHOD=$3
NORM=$4
MODEL=$5
FIXED=$6
DP=$7

CUDA_VISIBLE_DEVICES="${CUDA_NUM}" nohup python attack.py \
  --model="${MODEL}" \
  --input_dir=images \
  --output_dir="${OUTPUT_DIR}" \
  --norm="${NORM}" --method="${METHOD}" --fixed_const="${FIXED}" ${DP:+--$DP} --show_true --show_loss > "${OUTPUT_DIR}"/nohup.out &
