# Copyright (c) [2024] Haopeng Geng, The University of Tokyo
# MIT License (https://opensource.org/licenses/MIT)

# Evaluation of 2 matrixes using DTW, if the matrix is distribution, the distance should be bhattacharyya distance.

import os
import argparse
import numpy as np
import torch
from seq2seq_vc.utils import read_hdf5, write_hdf5, find_files

from tqdm import tqdm

# get KL divergence
# from scipy.stats import entropy
from fastdtw import fastdtw

def softmax(x):
    """Compute softmax values for matrix x."""
    e_x = np.exp(x - x.max(axis = 1).reshape(len(x), 1))
    return e_x / e_x.sum(axis=1).reshape(len(x), 1)

def euclid_dist(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def bhat_dist(v1, v2):
    """Bhattacharyya distance requires v1 and v2 to be sqrt."""
    return -np.log(np.maximum((v1 * v2).sum(), 1e-12))

def kl_dist(v1, v2, vl1, vl2):
    """Kullback-Leibler distance requires vl1 and vl2 to be the log value of v1 and v2."""
    return (v1 * (vl1 - vl2)).sum()

def calc_dist_bhat(p1, p2):
    #print(p1)
    p1 = np.sqrt(softmax(p1)).astype(np.float64)
    #print(p1)
    p2 = np.sqrt(softmax(p2)).T.astype(np.float64)
    dist = -np.log(np.matmul(p1,p2))
    #print(dist)
    return dist

from scipy.linalg import sqrtm
from fastdtw import fastdtw

# 计算DTW距离

# print(f"DTW distance between the two matrices using Bhattacharyya distance: {distance}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Reference matrix")
    parser.add_argument("--trg_dir", type=str, required=True, help="Hypothesis matrix")
    parser.add_argument("--feat", type=str, required=True, help="Feature type")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    # check directory existence
    assert os.path.exists(args.src_dir), f"Reference directory not found: {args.src_dir}"
    assert os.path.exists(args.trg_dir), f"Hypothesis directory not found: {args.trg_dir}"
    
    # iterate over files, get all surfix '.h5' files, they are feature files
    src_files = find_files(args.src_dir, query="*.h5", include_root_dir=True)
    trg_files = find_files(args.trg_dir, query="*.h5", include_root_dir=True)

    # sort by the basename of paths    
    src_files.sort(key=lambda x: x.split('/')[-1])
    trg_files.sort(key=lambda x: x.split('/')[-1])
    
    # check parallel
    assert len(src_files) == len(trg_files), f"Reference and Hypothesis files are not in parallel"
    
    # get id_list
    for src, trg in tqdm(zip(src_files, trg_files)):
        edit_distance = []
        try:
            assert src.split('/')[-1] == trg.split('/')[-1], f"Reference and Hypothesis files are not in parallel"
        except:
            import pdb; pdb.set_trace()
        id = src.split('/')[-1].split('.')[0]
        ref = read_hdf5(src, args.feat)
        hyp = read_hdf5(trg, args.feat)
        
        if args.feat == 'ppg_sxliu':
            try:
                dist = calc_dist_bhat(ref, hyp)
                result, path = fastdtw(ref, hyp, dist=bhat_dist)
                edit_score = result / ref.shape[0]
                edit_distance.append(edit_score)
                write_hdf5(os.path.join(args.output, f"{id}.h5"), 'dist', dist)
                write_hdf5(os.path.join(args.output, f"{id}.h5"), 'dtw_path', path)
                write_hdf5(os.path.join(args.output, f"{id}.h5"), 'edit_dis', edit_score)
            except:
                import pdb; pdb.set_trace()
        else:
            raise NotImplementedError(f"Feature type {args.feat} not implemented")
        
    edit_distance_mean = np.array(edit_distance).mean()
    print (f"Mean edit distance between {args.src_dir} and {args.src_dir}: {edit_distance_mean}")
    return edit_distance

if __name__ == "__main__":
    main()
    