# copy wav to dir base on scp file
import os
import sys
import shutil


wav_scp = sys.argv[1]
trg_dir = sys.argv[2]

if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

with open(wav_scp, 'r') as f:
    for line in f:
        line = line.strip()
        utt, wav = line.split(" ")
        wav_id = wav.split("/")[-1]
        shutil.copy(wav, os.path.join(trg_dir, wav_id))
        print('copy {} to {}'.format(wav, trg_dir))
