#!/usr/bin/env bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=4      # stage to start
stop_stage=4 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

conf=/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/conf/aas_vc.ppgmelppg.V000_R_V006_SS.yaml
# dataset configuration
src_db_root=./downloads/V000_R_max_valid    # default saved here
trg_db_root=./downloads/V006_SS_max_valid  # PLEASE CHANGE THIS
dumpdir=dump                                    # directory to dump full features
srcspk=V000_R_max_valid
trgspk=V006_SS_max_valid
num_train=2000 # out of 2696
num_dev=395
num_eval=300
stats_ext=h5
norm_name=ljspeech                              # used to specify normalized data.

# pretrained model related
pretrained_model_checkpoint=downloads/ljspeech_text_to_ppg_sxliu_aept/checkpoint-50000steps.pkl
npvc_checkpoint=downloads/s3prl-vc-ppg_sxliu/checkpoint-50000steps.pkl
npvc_name=ppg_sxliu

# fine-tuning related setting
transfer_learning_model_dir="/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/lsc/exp/V000_R_max_valid_V006_SS_max_valid_2000_vtn.tts_pt.v1.ppg_sxliu_V006"

# training related setting
tag=""     # tag for directory to save model
resume="/home/kevingenghaopeng/vc/seq2seq-vc/egs/l2-arctic/lsc/exp/V006_S1_max_valid_V006_SS_max_valid_2000_vtn.tts_pt.v1.ppg_sxliu_V006S1_to_V006SS_decoder/checkpoint-60000steps.pkl"  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
expdir="/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/exp/V000_R_max_valid_V006_SS_max_valid_2000_V000_R_to_V006_SS/"

checkpoint="${expdir}/checkpoint-55000steps.pkl"               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
decode_dumpdir="/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/dump/V000_R_max_valid_eval/norm_self"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

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
                --checkpoint "${checkpoint}" \
                --src-feat-type "${npvc_name}" \
                --trg-feat-type "${npvc_name}" \
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
