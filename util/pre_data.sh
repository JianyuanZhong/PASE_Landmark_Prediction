#!bin/bash

python data_prep.py --mfcc_root /home/cxu-serve/p1/lchen63/grid/mfcc --landmark_root /home/cxu-serve/p1/lchen63/grid/landmark --audio_root /home/cxu-serve/p1/lchen63/grid/oppo/audio\
	--dev 20 --test 30 --out_dir ../data/oppo_all.cfg
