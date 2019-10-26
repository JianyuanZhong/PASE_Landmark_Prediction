#!/bin/bash

python train.py --model LSTM\
	--PASE_lr 0.00001 --PASE_ckpt ../PASE_models/qrnn/PASE_QRNN_512-256.ckpt --PASE_cfg ../PASE_models/qrnn/qrnn_256.cfg --MLP_cfg cfg/LSTM.cfg\
	--PASE_optim 1 --lr 0.0001 --save_path ../PASE2landmark/LSTM_no_ref_3_layer_l1_loss_temp_mlp_out --batch_size 32\
	--context_left 0 --context_right 0 --landmark_norm False\
	--early_stopping False --add_ref True --epoch 30\
	--num_worker 2 --feature wav
