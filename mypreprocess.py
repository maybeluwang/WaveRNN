import os
import numpy as np
import math, pickle, os
from utils import hparams as hp
from utils import dsp
from utils import *
from tqdm import tqdm
import argparse
def convert_file(path):
    y = dsp.load_wav(path)
    mel = dsp.melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = dsp.encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = dsp.float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_data(wav_dir, output_dir, train_list, test_list):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")


    
    train_mel_path = os.path.join(train_dir,"mel")
    train_wav_path = os.path.join(train_dir,"quant")

    test_mel_path = os.path.join(test_dir,"mel")
    test_wav_path = os.path.join(test_dir,"quant")

    # create dirs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_mel_path, exist_ok=True)
    os.makedirs(train_wav_path, exist_ok=True)
    os.makedirs(test_mel_path, exist_ok=True)
    os.makedirs(test_wav_path, exist_ok=True)


    # get list of wav files
    if train_list == None and test_list == None:
        wav_files = os.listdir(wav_dir)
        # check wav_file
        assert len(wav_files) != 0 or wav_files[0][-4:] == '.wav', "no wav files found!"
        # create training and testing splits
        test_wav_files = wav_files[-3:]
        wav_files = wav_files[:-3]
    else:
        wav_files = get_wavfile_list(train_list)
        test_wav_files = get_wavfile_list(test_list)
    
    for file_id in wav_files:
        print(file_id)
        # get the file id
        mel, quant = convert_file(os.path.join(wav_dir,file_id+'.wav'))
        # save
        np.save(os.path.join(train_mel_path,file_id+".npy"), mel)
        np.save(os.path.join(train_wav_path,file_id+".npy"), quant)
     
    with open(os.path.join(train_dir,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(wav_files, f)
    

    # process testing_wavs
    for file_id in test_wav_files:
        mel, quant = convert_file(os.path.join(wav_dir, file_id+'.wav'))
        # save test_wavs
        np.save(os.path.join(test_mel_path,file_id+".npy"), mel)
        np.save(os.path.join(test_wav_path,file_id+".npy"), quant)


    with open(os.path.join(test_dir,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(test_wav_files, f)

    
    print("\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(wav_files), os.path.abspath(output_dir)))

def get_wavfile_list(path):
    with open(path, 'r') as f:
         all_lines = f.readlines()
    wav_files = [line.split('|')[0]  for line in all_lines]
    return wav_files

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--wav_dir', metavar='DIR')
parser.add_argument('--output_dir', metavar='DIR')
parser.add_argument('--train_list', metavar='FILE')
parser.add_argument('--test_list', metavar='FILE')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file

process_data(args.wav_dir, args.output_dir, args.train_list, args.test_list)



def test_get_wav_mel():
    wav, mel = get_wav_mel('sample.wav')
    print(wav.shape, mel.shape)
    print(wav)
