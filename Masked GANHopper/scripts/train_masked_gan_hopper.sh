set -ex
python train.py --dataroot ./datasets/RainyRoad --name RainyRoad_MGH --model masked_gan_hopper --no_dropout
