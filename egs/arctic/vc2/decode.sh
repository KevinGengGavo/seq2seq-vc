#!/usr/bin/env bash

# Copyright 2023 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=4      # number of parallel jobs in feature extraction

conf=conf/aas_vc.ppgmelppg.V000_R_V006_SS.yaml

# dataset configuration
db_root=./downloads
dumpdir=dump                # directory to dump full features
srcspk="V000_R_max_valid"                 # available speakers: "clb" "bdl"
trgspk="V006_S1_max_valid"                  # available speakers: "slt" "rms"
num_train=2000
stats_ext=h5
norm_name='self'                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

src_feat=ppg_sxliu
trg_feat=mel
dp_feat=ppg_sxliu

train_duration_dir=none     # need to be properly set if FS2-VC is used
dev_duration_dir=none       # need to be properly set if FS2-VC is used

# pretrained model related
pretrained_model_checkpoint=""
# "/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/downloads/ljspeech_text_to_ppg_sxliu_aept/checkpoint-50000steps.pkl"

# training related setting
tag="V000_SS_to_V006_S1"     # tag for directory to save model
resume=""
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
checkpoint=""
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# evaluation related setting
gv=False                    # whether to calculate GV for evaluation


# decoding related setting
expdir=exp/V006_SS_max_valid_V006_S1_max_valid_2000_V000_SS_to_V006_S1

checkpoint=${expdir}/checkpoint-100000steps.pkl               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
decode_dumpdir=dump/V000_R_max_valid_eval/norm_self

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

# set -euo pipefail


if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${srcspk}_eval"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        CUDA_VISIBLE_DEVICES="" ${cuda_cmd} JOB=1:${n_jobs} --gpu 0 "${outdir}/${name}/decode.JOB.log" \
            vc_decode.py \
                --dumpdir "${decode_dumpdir}/dump.JOB" \
                --dp_input_dumpdir "${decode_dumpdir}/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --src-feat-type "${src_feat}" \
                --trg-feat-type "${trg_feat}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi
