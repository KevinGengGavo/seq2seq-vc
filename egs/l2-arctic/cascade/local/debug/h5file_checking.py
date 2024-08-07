from parallel_wavegan.utils import read_hdf5
from seq2seq_vc.trainers.ar_vc import ARVCTrainer

from pathlib import Path
path = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/exp/TXHC_bdl_1032_stft_view_as_real/results/checkpoint-50000steps/stage2_ppg_sxliu_checkpoint-50000steps_TXHC_eval/mel/arctic_b0490.h5"

ppg = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/lsc/dump/TXHC_train_1032/ppg_sxliu/raw/dump.1/ppg_sxliu/arctic_a0001.h5" # (489, 144)
ppg_ = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/lsc/dump/TXHC_train_1032/ppg_sxliu/norm_ljspeech/dump.1/arctic_a0001.h5"
import pdb; pdb.set_trace()

x = read_hdf5(path, "mel")
y = read_hdf5(ppg, "ppg_sxliu")
z = read_hdf5(ppg_, "ppg_sxliu")

s3prl_stat = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/s3prl-vc-ppg_sxliu/stats.h5"
THXC_stat = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/pwg_TXHC/stats.h5"
cpy_stat = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/s3prl-vc-ppg_sxliu/stats copy.h5"

import pdb; pdb.set_trace()

f0_mean = read_hdf5(s3prl_stat, "mean")
f0_scale = read_hdf5(s3prl_stat, "scale")
THXC_mean = read_hdf5(THXC_stat, "mean")
THXC_scale = read_hdf5(THXC_stat, "scale")
f0_scale = read_hdf5(THXC_stat, "scale")