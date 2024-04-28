from parallel_wavegan.utils import read_hdf5
from pathlib import Path
path = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/exp/TXHC_bdl_1032_vtn.tts_pt.v1/results/checkpoint-50000steps/stage2_ppg_sxliu_checkpoint-50000steps_TXHC_dev/mel/arctic_b0440.h5"

x = read_hdf5(path, "mel")
import pdb; pdb.set_trace()

s3prl_stat = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/s3prl-vc-ppg_sxliu/stats.h5"
THXC_stat = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/THXC_v1/stats.h5"
f0_mean = read_hdf5(stat, "mean")
f0_scale = read_hdf5(stat, "scale")

import pdb; pdb.set_trace()