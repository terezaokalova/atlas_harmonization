#!/usr/bin/env python3
"""
This script averages the functional connectivity matrices for each subject.
For each connectivity method, it iterates over clips 1 to 20 in the subject's
"fc_matrices" folder, skips missing or problematic clips (e.g. matrices that are all NaN or zero),
stacks the valid matrices, and computes the nanmean over the clip axis.
The resulting averaged 2D fc matrix (n_chan x n_chan) is then saved in a subfolder
named "fc_matrices_averaged" inside the subject folder.
"""

import os
import pickle
import numpy as np
# from config.config import CLEAN_PATH

CLEAN_PATH = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"

# list of connectivity methods
fc_methods = ["pearson", "squared_pearson", "cross_corr", "coh", "plv", "rela_entropy"]

# Tolerance for considering a matrix to be all zero
ZERO_TOL = 1e-8

# Get all subject folders (names starting with "sub-") in the CLEAN_PATH directory
subject_dirs = sorted([d for d in os.listdir(CLEAN_PATH)
                       if os.path.isdir(os.path.join(CLEAN_PATH, d)) and d.startswith("sub-")])

for subject in subject_dirs:
    subject_path = os.path.join(CLEAN_PATH, subject)
    fc_folder = os.path.join(subject_path, "fc_matrices")
    avg_folder = os.path.join(subject_path, "fc_matrices_averaged")
    if not os.path.exists(fc_folder):
        print(f"[WARNING] fc_matrices folder not found for subject {subject}. Skipping subject.")
        continue
    if not os.path.exists(avg_folder):
        os.makedirs(avg_folder)
    
    print(f"\n=== Processing subject: {subject} ===")
    
    for method in fc_methods:
        valid_matrices = []
        valid_clips = []
        # Expecting clip indices 1 to 20 (files are named with clip number 1-indexed)
        for clip in range(1, 21):
            filename = f"{subject}_fc_{method}_clip{clip}.pkl"
            filepath = os.path.join(fc_folder, filename)
            if not os.path.exists(filepath):
                print(f"[INFO] {method}: file not found for clip {clip}")
                continue
            with open(filepath, "rb") as f:
                result = pickle.load(f)
            fc_matrix = result.get("fc_matrix")
            if fc_matrix is None:
                print(f"[WARNING] {method}: fc_matrix missing in clip {clip}")
                continue
            # Ensure matrix is 2D and square (n_chan x n_chan)
            if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
                print(f"[WARNING] {method}: unexpected shape {fc_matrix.shape} in clip {clip}")
                continue
            # Check if matrix is entirely NaN or nearly zero
            if np.all(np.isnan(fc_matrix)) or np.allclose(fc_matrix, 0, atol=ZERO_TOL):
                print(f"[WARNING] {method}: fc_matrix in clip {clip} is all NaN or zero")
                continue
            valid_matrices.append(fc_matrix)
            valid_clips.append(clip)
        
        if len(valid_matrices) == 0:
            print(f"[ERROR] No valid {method} fc matrices found for {subject}.")
            continue
        
        # Verify all valid matrices have the same shape
        shapes = [m.shape for m in valid_matrices]
        if len(set(shapes)) != 1:
            print(f"[ERROR] Inconsistent shapes for {method} in {subject}: {shapes}")
            continue
        
        # Average over the valid matrices (stacking along a new axis)
        stacked = np.stack(valid_matrices, axis=0)
        avg_matrix = np.nanmean(stacked, axis=0)
        # Use the channel list from the last successfully loaded result (assumed identical across clips)
        channels = result.get("channels", [])
        
        # Save the averaged fc matrix in the "fc_matrices_averaged" folder
        out_filename = f"{subject}_fc_{method}_averaged.pkl"
        out_filepath = os.path.join(avg_folder, out_filename)
        with open(out_filepath, "wb") as f:
            pickle.dump({"channels": channels, "fc_matrix": avg_matrix}, f)
        
        print(f"Averaged {method} for {subject} using clips {valid_clips} -> {out_filepath}")
