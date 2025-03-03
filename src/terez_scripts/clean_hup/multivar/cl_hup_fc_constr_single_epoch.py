#!/usr/bin/env python3
# name: cl_hup_fc_constr_single_epoch.py
# this script processes a single epoch for a given subject.
# it loads the epoch data, computes connectivity measures via the cnt tools in 5-second windows with no overlap,
# averages the outputs to a single 2d connectivity matrix for each method, and saves the results.

import os
import sys
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(0, "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/external_dependencies/CNT_research_tools/python/CNTtools")
sys.path.insert(0, "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization")

from cl_hup_data_loading import get_clean_hup_file_paths, load_epoch
from iEEGPreprocess import iEEGData 

from config.config import BASE_PATH
from config.config import CLEAN_PATH
base_path = BASE_PATH
clean_path = CLEAN_PATH
# clean_path = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean'

def reduce_connectivity_matrix(mat):
    """
    reduce the connectivity output to a 2d (n_channels x n_channels) matrix by 
    averaging only the time and frequency dimensions.
    expected cases:
      - if mat.shape == (n_windows, n_channels, n_channels, n_freq): average over axis 0 (time) and axis 3 (freq)
      - if mat.shape == (n_channels, n_channels, n_freq): average over frequency axis (axis=-1)
      - if mat is already (n_channels, n_channels), return as is.
    """
    if mat.ndim == 4:
        # average over time windows and frequency bins
        return np.mean(mat, axis=(0, 3))
    elif mat.ndim == 3:
        # assume the extra dimension is frequency; average over it
        return np.mean(mat, axis=-1)
    else:
        return mat

def process_epoch(subject, epoch_idx, clean_path, fc_methods, win_size_sec=5, overlap_sec=0):
    subject_path = os.path.join(clean_path, subject)
    subject_files = get_clean_hup_file_paths(clean_path)
    # since the epoch keys are 1-indexed (1,2,...,20), use epoch_idx+1
    file_path = subject_files[subject][epoch_idx + 1]
    df = load_epoch(file_path)
    ch_names = df.index.to_list()
    data = np.column_stack(df['data'].values)
    fs = 200  # each epoch is 1 minute at 200 Hz

    # create an iEEGData instance; no filtering applied here
    ieeg = iEEGData(
        filename=f"{subject}_clip{epoch_idx + 1}",
        start=0,
        stop=60,
        data=data,
        fs=fs,
        ch_names=ch_names
    )
    
    # compute connectivity measures using a windowed approach (5s windows, no overlap)
    ieeg.connectivity(
        methods=fc_methods,
        win=True,
        win_size=win_size_sec,
        segment=win_size_sec,
        overlap=overlap_sec
    )

    for method in fc_methods:
        conn_out = ieeg.conn[method]
        if isinstance(conn_out, tuple):
            conn_out = conn_out[0]
        # explicitly reduce to a 2d (n_channels x n_channels) matrix 
        mat = reduce_connectivity_matrix(conn_out)

        result = {"channels": ch_names, "fc_matrix": mat}
        # file naming: first clip becomes clip1
        out_filename = f"{subject}_fc_{method}_clip{epoch_idx + 1}.pkl"
        out_filepath = os.path.join(subject_path, out_filename)
        with open(out_filepath, "wb") as f:
            pickle.dump(result, f)
        print(f"saved {method} fc for {subject}, clip {epoch_idx + 1} -> {out_filepath}")

def main():
    if len(sys.argv) < 3:
        print("usage: python cl_hup_fc_constr_single_epoch.py <subject_folder> <epoch_index>")
        sys.exit(1)
    subject = sys.argv[1]
    epoch_idx = int(sys.argv[2])
    fc_methods = ["pearson", "squared_pearson", "cross_corr", "coh", "plv", "rela_entropy"]
    process_epoch(subject, epoch_idx, clean_path, fc_methods, win_size_sec=5, overlap_sec=0)

if __name__ == "__main__":
    main()

# fixing off-by-one error in naming of output files (logic was good)
# def process_epoch(subject, epoch_idx, clean_path, fc_methods, win_size_sec=5, overlap_sec=0):
#     subject_path = os.path.join(clean_path, subject)
#     subject_files = get_clean_hup_file_paths(clean_path)
#     file_path = subject_files[subject][epoch_idx + 1]  # dictionary key 1..20
#     df = load_epoch(file_path)
#     ch_names = df.index.to_list()
#     data = np.column_stack(df['data'].values)
#     fs = 200

#     ieeg = iEEGData(
#         filename=f"{subject}_clip{epoch_idx + 1}",
#         start=0,
#         stop=60,
#         data=data,
#         fs=fs,
#         ch_names=ch_names
#     )
    
#     ieeg.connectivity(
#         methods=fc_methods,
#         win=True,
#         win_size=win_size_sec,
#         segment=win_size_sec,
#         overlap=overlap_sec
#     )

#     for method in fc_methods:
#         conn_out = ieeg.conn[method]
#         if isinstance(conn_out, tuple):
#             conn_out = conn_out[0]
#         mat = conn_out
#         while mat.ndim > 2:
#             mat = np.mean(mat, axis=0)

#         result = {"channels": ch_names, "fc_matrix": mat}

#         # name the file as clip{epoch_idx + 1}
#         out_filename = f"{subject}_fc_{method}_clip{epoch_idx + 1}.pkl"
#         out_filepath = os.path.join(subject_path, out_filename)
#         with open(out_filepath, "wb") as f:
#             pickle.dump(result, f)

#         print(f"saved {method} fc for {subject}, clip {epoch_idx + 1} -> {out_filepath}")

# # def process_epoch(subject, epoch_idx, clean_path, fc_methods, win_size_sec=5, overlap_sec=0):
# #     subject_path = os.path.join(clean_path, subject)
# #     # load file paths for this subject
# #     subject_files = get_clean_hup_file_paths(clean_path)
# #     # since the epoch keys are 1-indexed (1,2,...,20), use (epoch_idx+1)
# #     file_path = subject_files[subject][epoch_idx + 1]
# #     df = load_epoch(file_path)
# #     ch_names = df.index.to_list()
# #     data = np.column_stack(df['data'].values)  # shape: (n_samples, n_channels)
# #     # each epoch is 1 minute (and already filtered), with sampling rate 200 hz
# #     fs = 200

# #     # create an iEEGData instance; no filtering applied here.
# #     ieeg = iEEGData(
# #         filename=f"{subject}_clip{epoch_idx}",
# #         start=0,
# #         stop=60,
# #         data=data,
# #         fs=fs,
# #         ch_names=ch_names
# #     )
    
# #     # compute connectivity measures using the provided methods.
# #     # using non-overlapping windows: overlap_sec set to 0.
# #     ieeg.connectivity(methods=fc_methods, win=True, win_size=win_size_sec, segment=win_size_sec, overlap=overlap_sec)

# #     # for each connectivity method, average extra dimensions to get a 2d matrix.
# #     for method in fc_methods:
# #         conn_out = ieeg.conn[method]
# #         if isinstance(conn_out, tuple):
# #             conn_out = conn_out[0]
# #         mat = conn_out
# #         while mat.ndim > 2:
# #             mat = np.mean(mat, axis=0)
# #         # store the final matrix and channel labels in a dict.
# #         result = {"channels": ch_names, "fc_matrix": mat}
# #         # save file with naming convention: sub-ridxxxx_fc_<method>_clip<epoch_idx>.pkl
# #         out_filename = f"{subject}_fc_{method}_clip{epoch_idx}.pkl"
# #         out_filepath = os.path.join(subject_path, out_filename)
# #         with open(out_filepath, "wb") as f:
# #             pickle.dump(result, f)
# #         print(f"saved {method} fc for {subject} epoch {epoch_idx} -> {out_filepath}")

# def main():
#     # base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
#     # get subject and epoch index from command line arguments
#     if len(sys.argv) < 3:
#         print("usage: python cl_hup_fc_constr_single_epoch.py <subject_folder> <epoch_index>")
#         sys.exit(1)
#     subject = sys.argv[1]
#     epoch_idx = int(sys.argv[2])
#     # define fc methods
#     fc_methods = ["pearson", "squared_pearson", "cross_corr", "coh", "plv", "rela_entropy"]
#     process_epoch(subject, epoch_idx, clean_path, fc_methods, win_size_sec=5, overlap_sec=0)

# if __name__ == "__main__":
#     main()