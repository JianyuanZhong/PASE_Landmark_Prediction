#!/bin/bash

python train.py \
	--PASE_lr 0.00001 --PASE_cfg ../PASE_models/qrnn/qrnn_256.cfg --PASE_ckpt ../PASE_models/qrnn/PASE_QRNN_512-256.ckpt --MLP_cfg cfg/MLP_linear.cfg\
	--PASE_optim 1 --lr 0.0001 --save_path ../PASE2landmark/mlp_qrnn_landmark_loss --batch_size 20\
	--context_left 0 --context_right 0 --landmark_norm False\
	--early_stopping False --add_ref True --epoch 30
