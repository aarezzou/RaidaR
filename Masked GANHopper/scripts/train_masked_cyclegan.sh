set -ex
python train.py --dataroot ./datasets/RainyRoad --name RainyRoad_MCG --model masked_gan_hopper --num_hops 1 --lambda_smooth 0 --lambda_hybrid 0 --no_dropout
