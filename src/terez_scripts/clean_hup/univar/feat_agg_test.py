#!/usr/bin/env python3
"""
Test script for aggregating feature tables for a single subject.

This script:
  - Loads all epoch feature tables (PKL files) for one subject.
  - Drops the raw 'data' column.
  - Separates metadata columns (assumed to be non-numeric or from a known set)
    from the computed feature columns.
  - Averages the feature columns across epochs (for each electrode).
  - Saves the aggregated table as <subject>_features_averaged.pkl and CSV
    in the subject folder.
  - Prints a summary of the aggregated DataFrame.

Usage:
    python test_aggregate_subject_features.py
"""

import os
import pandas as pd
import numpy as np

def aggregate_subject_features(subject_dir, output_dir=None):
    """
    Aggregate feature tables across epochs for a single subject.

    Parameters:
        subject_dir (str): Path to the subject folder (e.g., ".../sub-RID0031")
        output_dir (str): Optional; if provided, save the aggregated table there.

    Returns:
        aggregated_df (pd.DataFrame): Averaged feature table.
    """
    # List all PKL files matching the naming convention for epochs.
    pkl_files = sorted([f for f in os.listdir(subject_dir)
                        if f.startswith("metadata_and_features_epch") and f.endswith(".pkl")])
    if len(pkl_files) == 0:
        raise ValueError(f"No epoch feature files found in {subject_dir}.")
    
    # Load each epoch's DataFrame.
    df_list = []
    for fname in pkl_files:
        fpath = os.path.join(subject_dir, fname)
        df = pd.read_pickle(fpath)
        # Drop the raw 'data' column if it exists.
        if 'data' in df.columns:
            df = df.drop(columns=['data'])
        df_list.append(df)
    
    # --- Determine metadata vs. feature columns ---
    # Option 1: If you have a known set of metadata column names, you can use that.
    known_metadata = {'labels', 'x', 'y', 'z', 'roi', 'roiNum', 'spared'}
    metadata_cols = [col for col in df_list[0].columns if col in known_metadata or df_list[0][col].dtype == 'O']
    feature_cols = [col for col in df_list[0].columns if col not in metadata_cols]

    # Use the metadata from the first epoch (assumed identical across epochs).
    metadata_df = df_list[0][metadata_cols].copy()
    
    # --- Average feature columns across epochs ---
    feature_dfs = [df[feature_cols] for df in df_list]
    # Stack the feature data into a 3D array: (n_epochs, n_electrodes, n_features)
    stacked_features = np.stack([df.values for df in feature_dfs], axis=0)
    # Compute the mean over the epoch axis (axis=0).
    averaged_features = np.mean(stacked_features, axis=0)
    
    # Create a DataFrame for the averaged features using the same index as the original.
    averaged_feat_df = pd.DataFrame(averaged_features, columns=feature_cols, index=df_list[0].index)
    
    # Combine the metadata and averaged feature DataFrames.
    aggregated_df = pd.concat([metadata_df, averaged_feat_df], axis=1)
    
    # Save the aggregated DataFrame.
    subject_name = os.path.basename(subject_dir)
    if output_dir is None:
        output_dir = subject_dir
    pkl_path = os.path.join(output_dir, f"{subject_name}_features_averaged.pkl")
    csv_path = os.path.join(output_dir, f"{subject_name}_features_averaged.csv")
    aggregated_df.to_pickle(pkl_path)
    aggregated_df.to_csv(csv_path, index=True)
    print(f"Aggregated features saved for {subject_name}:")
    print(f"  PKL: {pkl_path}")
    print(f"  CSV: {csv_path}")
    
    return aggregated_df

# --- Test Code for a Single Subject ---
if __name__ == '__main__':
    # Set the absolute path to the base subject folder.
    base_subject_dir = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
    # For testing, choose one subject folder (e.g., sub-RID0031).
    subject_folder = os.path.join(base_subject_dir, "sub-RID0031")
    
    # Run the aggregation function for the selected subject.
    aggregated_df = aggregate_subject_features(subject_folder)
    
    # Print out a preview of the aggregated DataFrame.
    print("Aggregated DataFrame Head:")
    print(aggregated_df.head())
    print("Aggregated DataFrame shape:", aggregated_df.shape)
