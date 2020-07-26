# Improving Black-box Adversarial Attacks with a Transfer-based Prior

This repository contains the code for reproducing the experimental results of our paper [Improving Black-box Adversarial Attacks with a Transfer-based Prior](https://arxiv.org/pdf/1906.06919.pdf) (NeurIPS 2019).

## Reproducing the results

The results can be reproduced using the following command:

```
mkdir outputs
python attack.py --input_dir=images --output_dir=outputs --model=[inception-v3 | vgg-16 | resnet-50 | jpeg | random | denoiser] --norm=[l2 | linfty] --method=[uniform | biased | fixed_biased | average | fixed_average] [--fixed_const=A FLOAT NUMBER] [--dataprior] [--show_true] [--show_loss]
```

About the attack algorithm, `method=[uniform | biased | fixed_biased | average | fixed_average]` corresponds to RGF, P-RGF (\\lambda^\*), P-RGF (\\lambda=`fixed_const`), Averaging (\\mu^\*), Averaging (\\mu=`fixed_const`) (the averaging method is introduced in Appendix B) in the paper respectively, `--norm=[l2 | linfty]` indicates the norm setting of the attack, and `--dataprior` is the option to incorporate the data-dependent prior. We also prepared a script `run_attack.sh` for convenience. The log of experimental results is stored in the file `[output_dir]/logging`, which records results for each image as well as the overall attack success rate and average number of queries over successful attacks.

We provide 6 target models included in the experiment section of the paper, in which 3 are non-defensive and 3 are defensive:
* 'inception-v3': The Inception-V3 model (non-defensive). Model checkpoint: http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/inception_v3.ckpt.
* 'vgg-16': The VGG-16 model (non-defensive). Model checkpoint: http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/vgg_16.ckpt.
* 'resnet-50': The ResNet-50 model (non-defensive). Model checkpoint: http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/resnet_v1_50.ckpt.
* 'jpeg': The JPEG defense built on the Inception-V3 model (defensive). Model checkpoint: The same as the checkpoint of 'inception-v3'.
* 'random': The randomized defense built on the Inception-V3 model (defensive). Model checkpoint: The same as the checkpoint of 'inception-v3'.
* 'denoiser': The High-Level Representation Guided Denoiser (defensive). Model checkpoint: http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/denoise_res_015.ckpt, http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/denoise_inres_014.ckpt and http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/denoise_incepv3_012.ckpt.

Note that before running the command above, the user needs to download:
1. Model checkpoints. We use the ResNet-152 model as the surrogate model, so please download the model checkpoint from http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/checkpoints/resnet_v2_152.ckpt. Then please download checkpoints of the target models you want to attack specified above. The downloaded checkpoints should be put under this folder.
2. The 1,000 images for evaluation from the validation set of ImageNet, which is available at http://ml.cs.tsinghua.edu.cn/~shuyu/p-rgf/images.zip. Then the user needs to unzip the downloaded files, and put the `images` folder under this folder such that the image files are directly under `images` folder (i.e., an image file should be at `./images/ILSVRC2012_val_00000019.png` instead of `./images/images/ILSVRC2012_val_00000019.png`).

## Requirements

Python packages: See `requirements.txt`.

The code is tested under Ubuntu 16.04/18.04, Python 3.6, CUDA 10.0 and cuDNN 7.5.0.
