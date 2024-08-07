#!/usr/bin/env bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=5      # stage to start
stop_stage=5 # stage to stop
verbose=5      # verbosity level (lower is less info)
n_gpus=5       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# dataset configuration
src_db_root=./downloads/V006_S1_max_valid    # default saved here
trg_db_root=./downloads/V006_SS_max_valid  # PLEASE CHANGE THIS
dumpdir=dump                                    # directory to dump full features
srcspk=V006_S1_max_valid
trgspk=V006_SS_max_valid
num_train=2000 # out of 2696
num_dev=395
num_eval=300
stats_ext=h5
norm_name=ljspeech                              # used to specify normalized data.

outdir="/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/exp/V000_R_max_valid_V006_SS_max_valid_2000_aas_vc.ppgppgppg.V006_SS_V006_S1/results/checkpoint-100000steps/V000_R_max_valid_eval/"
trg_db_root="data/V006_S1_max_valid_eval"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

# set -euo pipefail

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Objective Evaluation"
    # if "S1" in trg_db_root, then use S1 as trgspk, and name evaluation.log as it in evaluation.S1.log
    #$ else if it's "SS", then use SS as trgspk, and name evaluation.log as it in evaluation.SS.log
    name=$(basename "${outdir}")
    
    echo "Evaluation start. See the progress via ${outdir}/evaluation.s1.log"
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/evaluation.s1.log" \
    local/evaluate_gavo.py \
        --wavdir "${outdir}" \
        --data_root "${trg_db_root}" \
        --trgspk ${trgspk} \
        --f0_path "conf/f0.yaml" 
fi