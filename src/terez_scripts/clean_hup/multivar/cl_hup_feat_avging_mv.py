#!/usr/bin/env python3
"""
Aggregate feature tables across epochs for each subject in the 'clean' directory.
For each subject (sub-XXXX) folder:
  - Loads all 20 epoch feature tables (PKL files).
  - Drops the raw 'data' column if present.
  - Identifies metadata columns up to and including 'spared'.
  - Everything after 'spared' is treated as numeric feature columns to average across epochs.
  - Averages each feature across the 20 epoch DataFrames, row by row.
  - Saves the aggregated table as <subject>_features_averaged.pkl and CSV in the subject folder.
  
Usage:
    python aggregate_all_subjects.py
"""

import os
import pandas as pd
import numpy as np
import pycatch22

def get_feature_list():
    # Bandpower features for each band in the spectral config.
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_features = [f"{band}_{metric}" for band in band_names for metric in ['power', 'rel', 'log']]
    
    # FOOOF features.
    fooof_features = [
        'fooof_aperiodic_offset', 
        'fooof_aperiodic_exponent', 
        'fooof_r_squared', 
        'fooof_error', 
        'fooof_num_peaks'
    ]
    
    # Entropy feature.
    entropy_features = ['entropy_5secwin']
    
    # Obtain catch22 feature names via a dummy call.
    dummy = np.random.randn(100).tolist()
    res = pycatch22.catch22_all(dummy, catch24=False)
    catch22_features = [f"catch22_{nm}" for nm in res['names']]
    
    return band_features + fooof_features + entropy_features + catch22_features

# Explicit list of features to average
FEATURES_TO_AVERAGE = get_feature_list()

def aggregate_subject_features(subject_dir):
    # Find all epoch PKL files in the subject folder.
    pkl_files = sorted(
        f for f in os.listdir(subject_dir)
        if f.startswith("metadata_and_features_epch") and f.endswith(".pkl")
    )
    if not pkl_files:
        print(f"No epoch feature files found in {subject_dir}. Skipping.")
        return None

    # Load each epoch DataFrame and drop the 'data' column if it exists.
    df_list = []
    for fname in pkl_files:
        fpath = os.path.join(subject_dir, fname)
        df = pd.read_pickle(fpath)
        if 'data' in df.columns:
            df = df.drop(columns=['data'])
        df_list.append(df)
    
    # Ensure all DataFrames have the same index.
    common_index = df_list[0].index
    df_list = [df.reindex(common_index) for df in df_list]
    
    # Determine metadata columns: all columns not in the explicit feature list.
    metadata_cols = [col for col in df_list[0].columns if col not in FEATURES_TO_AVERAGE]
    
    # Extract the feature columns from each epoch.
    feature_dfs = []
    for df in df_list:
        # Some epochs might be missing a feature column; fill missing ones with NaN.
        feat_df = df.reindex(columns=FEATURES_TO_AVERAGE)
        feature_dfs.append(feat_df)
    
    # Stack into a 3D array: (n_epochs, n_electrodes, n_features)
    stacked_features = np.stack([fdf.values for fdf in feature_dfs], axis=0)
    # Average across epochs (axis=0)
    averaged_features = np.nanmean(stacked_features, axis=0)
    
    averaged_feat_df = pd.DataFrame(
        averaged_features,
        index=common_index,
        columns=FEATURES_TO_AVERAGE
    )
    
    # Use metadata from the first epoch.
    metadata_df = df_list[0][metadata_cols].copy()
    aggregated_df = pd.concat([metadata_df, averaged_feat_df], axis=1)
    
    subject_name = os.path.basename(subject_dir)
    out_pkl = os.path.join(subject_dir, f"{subject_name}_features_averaged.pkl")
    out_csv = os.path.join(subject_dir, f"{subject_name}_features_averaged.csv")
    aggregated_df.to_pickle(out_pkl)
    aggregated_df.to_csv(out_csv, index=True)
    
    print(f"[{subject_name}] Aggregated features saved to:")
    print(f"  {out_pkl}")
    print(f"  {out_csv}")
    
    return aggregated_df

# def main():
#     base_clean_dir = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
        
#     single_subject_folder = os.path.join(base_clean_dir, "sub-RID0031")
#     aggregate_subject_features(single_subject_folder)

def main():
    # Path to the 'clean' directory containing sub-RIDXXXX folders
    base_clean_dir = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"

    # Find all subject folders
    subject_folders = sorted(
        os.path.join(base_clean_dir, d)
        for d in os.listdir(base_clean_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(base_clean_dir, d))
    )

    # Aggregate for each subject
    for subj_folder in subject_folders:
        print(f"Aggregating for {subj_folder} ...")
        aggregate_subject_features(subj_folder)

if __name__ == "__main__":
    main()
