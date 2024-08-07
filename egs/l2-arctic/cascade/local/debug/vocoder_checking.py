import torch
import logging
import time
import torch
import yaml
import pdb
from parallel_wavegan.utils import load_model

config = "/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/cascade/downloads/s3prl-vc-ppg_sxliu/config.yml"
with open(config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
    checkpoint = config['vocoder']['checkpoint']
    stats = config['vocoder']['stats']
    
    # model = load_model(checkpoint, config)
    pdb.set_trace()
    model = torch.load(checkpoint)

