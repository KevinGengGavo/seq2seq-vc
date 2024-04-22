#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=50
num_eval=50
num_train=591
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/cms_us_slt_arctic slt data"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=100)."
    echo "    --num_eval: number of evaluation uttreances (default=100)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

# check speaker
available_spks=(
    "V000" "V001_SS" "V001_S1" "V001_S2"
)
if ! echo "${available_spks[*]}" | grep -q "${spk}"; then
    echo "Specified speaker ${spk} is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"
segments="${data_dir}/all/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp
find "$(realpath ${db_root})" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> "${scp}"
done

# make segments
# NOTE: this VAD only works for 16kHz!
python local/vad.py --scp_file "${scp}" \
                    --vad_dir "${data_dir}/all/vad/" \
                    --n_workers 16 \
                    --segments_path "${data_dir}/all/segments" \

# check
diff -q <(awk '{print $1}' "${scp}") <(awk '{print $1}' "${segments}") > /dev/null

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train_temp=$((num_all - num_deveval))
utils/split_data.sh \
    --num_first "${num_train_temp}" \
    --num_second "${num_deveval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}_temp" \
    "${data_dir}/deveval"
utils/split_data.sh \
    --num_first "${num_dev}" \
    --num_second "${num_eval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/deveval" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${eval_set}"

# check if further splitting is necessary
num_train_temp2=$((num_train_temp - num_train))
if [ ${num_train_temp2} -gt 0 ]; then
    utils/split_data.sh \
        --num_first "${num_train}" \
        --num_second "${num_train_temp2}" \
        --shuffle "${shuffle}" \
        "${data_dir}/${train_set}_temp" \
        "${data_dir}/${train_set}" \
        "${data_dir}/${train_set}_temp2"
    rm -rf "${data_dir}/${train_set}_temp"
    rm -rf "${data_dir}/${train_set}_temp2"
elif [ ${num_train_temp2} -eq 0 ]; then
    mv "${data_dir}/${train_set}_temp" "${data_dir}/${train_set}"
else
    echo "Please make sure num_train (${num_train}) + num_dev (${num_dev}) + num_eval (${num_eval}) = 691."
    exit 1
fi

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/deveval"

echo "Successfully prepared data."
