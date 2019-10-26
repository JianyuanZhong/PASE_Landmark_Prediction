#!/bin/bash

python train.py \
	--PASE_lr 0.000001 --PASE_ckpt ../PASE_models/MATconv/FE_e62.ckpt --MLP_cfg cfg/MLP_sigmoid.cfg\
	--PASE_optim 1 --lr 0.0005 --save_path ../PASE2landmark/img_bce_test --batch_size 32\
	--context_left 1 --context_right 1
