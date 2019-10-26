import sys, os
import argparse
import numpy as np
import json
from tqdm import tqdm


def get_spks_info(args):
    spk_lst = os.listdir(args.audio_root)

    count_wav = 0
    count_landmark = 0
    count_mfcc = 0
    wav_dict = {}
    landmark_dict = {}
    mfcc_dict = {}
    pbar = tqdm(spk_lst)
    for j, spk in enumerate(pbar):

        landmark_list = os.listdir(os.path.join(args.landmark_root, spk))

        wav_files = []
        landmark_files = []
        mfcc_files = []
        for file in landmark_list:
            if "norm" not in file:
                file_name = file.split('_')[0]

                # add landmark file to list
                landmark_files.append(file)

                # check if corresponding wav file exit. If exitst, add to list 
                if os.path.isfile(os.path.join(args.audio_root, spk, file_name) + '.wav'):
                    wav_files.append(file_name + '.wav')
                else:
                    raise ValueError("wav file not exist!!")

                if os.path.isfile(os.path.join(args.mfcc_root, spk, file_name) + '_mfcc.npy'):
                    mfcc_files.append(file_name + '_mfcc.npy')
                else:
                    raise ValueError("mfcc file not exist!!")


                

        count_wav += len(wav_files)
        count_landmark += len(landmark_files)
        count_mfcc += len(mfcc_files)
        wav_dict[spk] = wav_files
        landmark_dict[spk] = landmark_files
        mfcc_dict[spk] = mfcc_files

    print("=" * 50)
    print("Number of speaker found: {}".format(len(spk_lst)))
    print("Number of wav files found: {}".format(count_wav))
    print("Number of landmark files found: {}".format(count_landmark))
    print("Number of mfcc files found: {}".format(count_mfcc))
    print("=" * 50)

    return wav_dict, landmark_dict, mfcc_dict

def dict_to_list(wav_dict, landmark_dict, mfcc_dict):

    if wav_dict is None or landmark_dict is None:
        return None, None

    universal_wav_lst = []
    universal_landmark_lst = []
    universal_mfcc_lst = []
    for spk, wav_lst in wav_dict.items():
        for i, wav in enumerate(wav_lst):
            universal_wav_lst.append(os.path.join(spk, wav))
            universal_landmark_lst.append(os.path.join(spk, landmark_dict[spk][i]))
            universal_mfcc_lst.append(os.path.join(spk, mfcc_dict[spk][i]))

    # check the indice in wav list and landmarks are aligned
    print("checking alignment of indice.....")
    
    for i, wav in enumerate(universal_wav_lst):
        # print(wav, universal_landmark_lst[i], universal_mfcc_lst[i])
        if wav.split('.')[0] != universal_landmark_lst[i].split('.')[0] and universal_landmark_lst[i].split('_')[0] != universal_mfcc_lst[i].split('_')[0]:
            raise ValueError("indice not aligned!!!")
    print("done!")

    return universal_wav_lst, universal_landmark_lst, universal_mfcc_lst


def main(args):
    wav_dict, landmark_dict, mfcc_dict = get_spks_info(args)

    if args.dev:
        print("=" * 50)
        print("leaving s{} out for dev set".format(args.dev))
        print("=" * 50)
        dev_wav_dict = {"s{}".format(args.dev) : wav_dict.pop("s{}".format(args.dev))}
        dev_landmark_dict = {"s{}".format(args.dev) : landmark_dict.pop("s{}".format(args.dev))}
        dev_mfcc_dict = {"s{}".format(args.dev) : mfcc_dict.pop("s{}".format(args.dev))}

    if args.test:
        print("=" * 50)
        print("leaving s{} out for test set".format(args.test))
        print("=" * 50)
        test_wav_dict = {"s{}".format(args.test) : wav_dict.pop("s{}".format(args.test))}
        test_landmark_dict = {"s{}".format(args.test) : landmark_dict.pop("s{}".format(args.test))}
        test_mfcc_dict = {"s{}".format(args.test) : mfcc_dict.pop("s{}".format(args.test))}

    print("generating trainset....")
    train_wavs, train_landmarks, train_mfcc = dict_to_list(wav_dict, landmark_dict, mfcc_dict)
    print("generating devset....")
    dev_wavs, dev_landmarks, dev_mfcc = dict_to_list(dev_wav_dict, dev_landmark_dict, dev_mfcc_dict)
    print("generating testset....")
    test_wavs, test_landmarks, test_mfcc = dict_to_list(test_wav_dict, test_landmark_dict, test_mfcc_dict)

    data_cfg = {
        'train' : {'wav' : train_wavs, 'landmark' : train_landmarks, 'mfcc': train_mfcc},
        'dev' : {'wav' : dev_wavs, 'landmark' : dev_landmarks, 'mfcc': dev_mfcc},
        'test' : {'wav' : test_wavs, 'landmark' : test_landmarks, 'mfcc':test_mfcc},
        'audio_root' : args.audio_root,
        'landmark_root' : args.landmark_root,
        'mfcc_root': args.mfcc_root
    }

    output_dir = '/'.join(args.out_dir.split('/')[:-1])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(args.out_dir, "w") as out:
        json.dump(data_cfg, out)

    print("done!")
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mfcc_root', type=str, help='root dir for mfcc files')
    parser.add_argument('--audio_root', type=str, help="audio directory of dataset")
    parser.add_argument('--landmark_root', type=str, help="landmark root directory")
    parser.add_argument('--dev', type=int, help="option for generating dev files")
    parser.add_argument('--test', type=int, help="option for generating test sets")
    parser.add_argument('--out_dir', type=str, help="output files directory")

    args = parser.parse_args()
    print(args)
    main(args)