#!/bin/zsh

python make_vedio.py --PASE_cfg ../PASE_models/qrnn/qrnn_256.cfg --PASE_ckpt ../PASE2landmark/LSTM_no_ref_3_layer_l1_loss_temp_mlp_out/weights_PASE-PASE-3.ckpt \
	--MLP_cfg cfg/LSTM.cfg --model_ckpt ../PASE2landmark/LSTM_no_ref_3_layer_l1_loss_temp_mlp_out/weights_PASE-PASE-3.ckpt --out_dir results --model LSTM
