<br><br><br>

# Masked GANHopper in PyTorch

We provide PyTorch implementation for Masked GANHopper and Masked CycleGAN based on the [pytorch CycleGAN implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Installation

- Install [PyTorch](http://pytorch.org) 1.5.0+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`.

## Dataset Format
- The dataset should have the following folders:
```bash
--testA
--testA_Mask
--testB
--testB_Mask
--trainA
--trainA_mask
--trainB
--trainB_mask
```
- Images must be in PNG format. For each image named [NAME].png in the folder [PHASEX] there should be a segmentation mask named [NAME].png in the folder [PHASEX]_mask. 
- The masks should have 1 channel with value in the range [0, n_labels). 
- Please note that the variale rec_weights should be n_labels values, comma separated. (rec_weights represents lambda in the paper)

## Training
```bash
#!./scripts/train_masked_gan_hopper.sh
python train.py --dataroot [PATH_TO_DATASET] --name [MODEL_NAME] --model masked_gan_hopper
```
- In order to use Masked CycleGAN, you can simply set the number of hops to 1 and the weight for the hybrid loss and the smoothness loss to 0:
```bash
#!./scripts/train_masked_cyclegan.sh
python train.py --dataroot [PATH_TO_DATASET] --name [MODEL_NAME] --model masked_gan_hopper --num_hops 1 --lambda_smooth 0 --lambda_hybrid 0
```
- The checkpoints, sample images, loss info, etc. will be saved to here: `./checkpoints/[MODEL_NAME]`.
## Testing

```bash
#!./scripts/test_masked_gan_hopper.sh
python test.py --dataroot [PATH_TO_DATASET] --name [MODEL_NAME] --model test --model_suffix [A or B]
```
- If you used Masked CycleGAN for training, remember to set the number of hops to 1
```bash
#!./scripts/test_masked_cyclegan.sh
python test.py --dataroot [PATH_TO_DATASET] --name [MODEL_NAME] --model test --model_suffix [A or B] --num_hops 1
```
- The test results will be saved to a html file here: `./results/[MODEL_NAME]/latest_test/index.html`.
