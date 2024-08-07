src_dir=$1
trg_dir=$2
norm=$3
feat=$4
tag=$5

# if tag!=None
if [ -n "$tag" ]; then
    output_dir=${src_dir}/dtw/${tag}
else
    output_dir=${src_dir}/dtw/$(basename ${trg_dir})
fi

python local/dtw_evaluation.py \
       --src_dir ${src_dir}/${feat}/${norm} \
       --trg_dir ${trg_dir} \
       --feat ${feat} \
       --output ${output_dir}