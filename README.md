# Improving Black-box Adversarial Attacks with a Transfer-based Prior

This repository contains the code for reproducing the experimental results of attacking Inception-v3 model, of our submission: *Improving Black-box Adversarial Attacks with a Transfer-based Prior*.

## Reproducing the results

The results can be reproduced using the following command:

```
mkdir outputs
python attack.py --input_dir=images --output_dir=outputs --checkpoint_path=inception_v3.ckpt --norm=[l2 | linfty] --method=[uniform | biased | fixed_biased] [--dataprior] [--show_loss]
```

About the attack algorithm, `method=[uniform | biased | fixed_biased]` corresponds to RGF, P-RGF (\\lambda^\*), P-RGF (\\lambda=0.5) in the paper respectively, `--norm=[l2 | linfty]` indicates the norm setting of the attack, and `--dataprior` is the option to incorporate the data-dependent prior. We also prepared a script `run_attack.sh` for convenience. The log of experimental results is stored in the file `[output_dir]/logging`, which records results for each image as well as the overall attack success rate and average number of queries over successful attacks.

Note that before running the command above, the user needs to download the model checkpoints of Inception-V3 and ResNet-V2-152 first, whose links are available at https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models. Then the user needs to unzip the downloaded files, and put `inception_v3.ckpt` and `resnet_v2_152.ckpt` under this folder.

## Requirements

Python packages: numpy, scipy, tensorflow-gpu, opencv-python.

The code is tested under Ubuntu 16.04, Python 3.6, Tensorflow 1.13.1, NumPy 1.14, CUDA 10.0 and cuDNN 7.5.0.
