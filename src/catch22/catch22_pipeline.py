#!/usr/bin/env python3
"""
Standalone script to load and preprocess (both HUP and MNI),
then compute catch22 features on the resulting time series data.

Requirements:
    pip install pycatch22 scipy numpy pandas

Usage:
    python catch22_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, Tuple, Set, Optional
import pycatch22

###############################################################################
# 1. PATHS & CONSTANTS
###############################################################################
# TODO: Triple-check these paths
DATA_BASE_PATH  = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data"
HUP_ATLAS_PATH  = os.path.join(DATA_BASE_PATH, "hup_atlas.mat")
HUP_DF_PATH     = os.path.join(DATA_BASE_PATH, "hup_df.csv")
MNI_ATLAS_PATH  = os.path.join(DATA_BASE_PATH, "mni_atlas.mat")
MNI_DF_PATH     = os.path.join(DATA_BASE_PATH, "mni_df.csv")
META_DATA_PATH  = os.path.join(DATA_BASE_PATH, "metaData.csv") 
# META_DATA_PATH  = os.path.join(DATA_BASE_PATH, "metadata.csv") 

# HUP vs. MNI column name differences
HUP_TS_COL      = "wake_clip"  # e.g. "wake_clip"
MNI_TS_COL      = "Data_W"     # e.g. "Data_W"
HUP_PATIENT_COL = "patient_no"
MNI_PATIENT_COL = "Patient"    # or "patient_no", if that is correct for your MNI data

###############################################################################
# 2. OUTCOME-FILTER HELPERS
###############################################################################
def load_and_process_outcomes(meta_data_path: str, hup_atlas_path: str) -> Tuple[Dict[int, bool], Set[int]]:
    """
    load metadata and create a "good outcome" mapping for HUP.
    'Engel_12_mo' and 'Engel_24_mo' are columns in the CSV which
    determine if a patient is good outcome (<1.3).

    Returns:
        (outcome_dict, good_patients_set)
            outcome_dict[patient_no] = True/False for 'good' vs 'bad' outcome
            good_patients_set = set of patient_nos that have 'good' outcome
    """
    # Load metadata CSV
    meta_df = pd.read_csv(meta_data_path)

    # Load HUP atlas to see which patients exist
    hup_atlas = sio.loadmat(hup_atlas_path)
    if 'patient_no' not in hup_atlas:
        raise KeyError("No 'patient_no' found in HUP atlas—can't map outcomes.")

    patient_nos_in_hup = np.unique(hup_atlas['patient_no'].flatten())

    outcomes_dict   = {}
    good_outcomes   = set()

    # row-wise parallel:
    #   row i -> patient i
    #   define "good" if max(Engel_12_mo, Engel_24_mo) < 1.3
    for i, row in meta_df.iterrows():
        if i >= len(patient_nos_in_hup):
            # If CSV is bigger than # of patients in the .mat
            print(f"CSV is bigger than # of patients in the .mat")
            break

        patient_no   = patient_nos_in_hup[i]
        max_engel    = max(row['Engel_12_mo'], row['Engel_24_mo'])
        is_good      = (max_engel < 1.3)
        outcomes_dict[patient_no] = is_good
        if is_good:
            good_outcomes.add(patient_no)

    print(f"[load_and_process_outcomes] Found {len(good_outcomes)} good-outcome patients out of {len(patient_nos_in_hup)} total.")
    return outcomes_dict, good_outcomes


def filter_by_outcome_and_build_good_indices(
    patient_nums: np.ndarray,
    good_outcome_patients: Set[int]
) -> np.ndarray:
    """
    # TODO: check if the patient-electrode mappings are intact
    From an array of patient numbers (one per electrode), return the
    subset of indices belonging to 'good' outcome patients.

    Args:
        patient_nums: shape (#electrodes,) array of patient IDs for each electrode
        good_outcome_patients: set of patient numbers with good outcomes

    Returns:
        good_indices (1D np.array of electrode indices to keep)
    """
    good_idx_list = []
    for i, pat_no in enumerate(patient_nums):
        if pat_no in good_outcome_patients:
            good_idx_list.append(i)
    return np.array(good_idx_list, dtype=int)

###############################################################################
# 3. ELECTRODE-FILTER HELPERS
###############################################################################
def get_good_electrodes_resected_soz(atlas: Dict) -> np.ndarray:
    """
    Identify 'good' electrodes that are NOT resected, NOT in SOZ.
    Returns a 1D array (indices) of these 'good' electrodes.
    """
    if not all(k in atlas for k in ['resected_ch', 'soz_ch']):
        print("Warning: 'resected_ch' or 'soz_ch' not found in atlas. Returning ALL electrodes.")
        n_ch = atlas['wake_clip'].shape[0] if 'wake_clip' in atlas else 0
        return np.arange(n_ch)

    resected = atlas['resected_ch'].flatten().astype(bool)
    soz      = atlas['soz_ch'].flatten().astype(bool)
    good_mask = ~(resected | soz)  # i.e., keep electrodes that are not resected nor in SOZ
    return np.where(good_mask)[0]


###############################################################################
# 4. LOADING & PREPROCESSING FOR A COHORT
###############################################################################
def load_and_preprocess_cohort(prefix: str,
                               atlas_path: str,
                               region_path: str,
                               ts_col: str,
                               patient_col: str,
                               good_outcome_patients: Optional[Set[int]] = None,
                               filter_good_elec: bool = True
                               ) -> pd.DataFrame:
    """
    Load the {prefix}_atlas.mat, extracts time series from `ts_col`,
    optionally filter out resected/SOZ electrodes, and also optionally filter
    out bad-outcome patients.

    Steps:
      # TODO: check if everything is intact
      1) Load .mat, .csv (CSV is loaded but not used in this minimal example—unless you have region usage).
      2) Ensure time series shape is (num_samples, num_electrodes).
         - If shape[0] < shape[1], we transpose. Print out booleans for clarity.
      3) If prefix == 'hup' and filter_good_elec=True, remove resected & SOZ electrodes.
      4) If good_outcome_patients is provided, remove electrodes that belong to bad-outcome patients.
      5) Return a DataFrame with columns = electrodes, rows = time points.

    Returns:
        df_ts: DataFrame, shape (#time_points, #kept_electrodes).
    """
    print(f"\n--- Loading {prefix.upper()} data ---")
    atlas_dict = sio.loadmat(atlas_path)  # load .mat
    region_df  = pd.read_csv(region_path) # load region CSV (not used here, but keep it for completeness)

    # Ensure `ts_col` is in the atlas
    if ts_col not in atlas_dict:
        raise KeyError(f"Column '{ts_col}' not found in {prefix} atlas data.")
    ts_data = atlas_dict[ts_col]

    # Print original shape, check if we need transpose
    print(f"  Original shape from '{ts_col}' is {ts_data.shape}")
    need_transpose = (ts_data.shape[0] < ts_data.shape[1])
    print(f"  (Check) Is #rows < #columns? => {need_transpose}")
    if need_transpose:
        ts_data = ts_data.T
        print(f"  Transposed shape is now {ts_data.shape} (rows=samples, cols=electrodes)")

    # If we want to also map each electrode to a patient, we can read from e.g. atlas_dict[patient_col].
    if patient_col not in atlas_dict:
        raise KeyError(f"Patient column '{patient_col}' not found in {prefix} atlas data.")
    all_patient_nums = atlas_dict[patient_col].flatten()  # shape (#electrodes,)

    # 1) Possibly remove resected/SOZ electrodes (mainly HUP)
    resected_soz_mask = np.arange(ts_data.shape[1])  # by default, keep all
    if prefix == 'hup' and filter_good_elec:
        resected_soz_mask = get_good_electrodes_resected_soz(atlas_dict)
        print(f"  -> HUP: filtering out resected/SOZ. #good_elec = {len(resected_soz_mask)} out of {ts_data.shape[1]}")

    # 2) Possibly remove bad-outcome electrodes (i.e., keep only good-outcome).
    #    This is done only if good_outcome_patients is provided.
    outcome_mask = np.arange(ts_data.shape[1])  # keep all by default
    if good_outcome_patients is not None:
        # Build array of shape (#electrodes,) for the patient no. each electrode belongs to.
        if len(all_patient_nums) != ts_data.shape[1]:
            raise ValueError("Mismatch: # of patient IDs != # of electrodes. Check your data shapes.")
        outcome_mask = filter_by_outcome_and_build_good_indices(all_patient_nums, good_outcome_patients)
        print(f"  -> Filtering out bad-outcome. #good_outcome_elec = {len(outcome_mask)} out of {ts_data.shape[1]}")

    # Intersection of the two sets of electrode indices
    final_mask = sorted(list(set(resected_soz_mask).intersection(set(outcome_mask))))

    # Subset the time-series data columns (electrodes) to final set
    ts_data = ts_data[:, final_mask]
    # Also subset patient array
    patient_nums_kept = all_patient_nums[final_mask]

    # Build a DataFrame: columns are electrode_i
    electrode_ids = [f"{prefix}_elec_{i}" for i in range(ts_data.shape[1])]
    df_ts = pd.DataFrame(ts_data, columns=electrode_ids)
    print(f"  Final shape for {prefix.upper()} time series after all filtering => {df_ts.shape}")

    # (If needed, you can store the 'patient_no' for each column, but in this example we skip.)

    return df_ts


###############################################################################
# 5. MAIN: LOAD BOTH COHORTS, COMPUTE CATCH22
###############################################################################
def main():
    """
    1) Load outcome info for HUP, get the set of 'good outcome' patients.
    2) Load + preprocess HUP:
       - Filter resected/SOZ electrodes
       - Filter out bad-outcome patients
    3) Load + preprocess MNI:
       - Possibly skip resected/SOZ if not relevant
       - Possibly skip outcome filter if not relevant
    4) Compute catch22 features and output
    """

    # -- (A) Load outcome info (for HUP) and get the set of 'good outcome' patients
    outcomes_dict, good_outcomes_set = load_and_process_outcomes(META_DATA_PATH, HUP_ATLAS_PATH)

    # -- (B) Load + preprocess HUP
    hup_df = load_and_preprocess_cohort(
        prefix='hup',
        atlas_path=HUP_ATLAS_PATH,
        region_path=HUP_DF_PATH,
        ts_col=HUP_TS_COL,
        patient_col=HUP_PATIENT_COL,
        good_outcome_patients=good_outcomes_set,  # filter out 'bad' outcome
        filter_good_elec=True                     # also filter resected/SOZ
    )

    # -- (C) Load + preprocess MNI
    # Assume we do NOT have outcome info for MNI, or we choose not to filter by outcome for MNI.
    # Also assume no resected/SOZ data for MNI, or we don't want to filter them.
    mni_df = load_and_preprocess_cohort(
        prefix='mni',
        atlas_path=MNI_ATLAS_PATH,
        region_path=MNI_DF_PATH,
        ts_col=MNI_TS_COL,
        patient_col=MNI_PATIENT_COL,
        good_outcome_patients=None,   # i.e. do not filter by outcome for MNI
        filter_good_elec=False        # assume we skip resected/SOZ for MNI
    )

    # We now have two DataFrames: hup_df and mni_df
    # Each is (#time_points, #filtered_electrodes)

    ###########################################################################
    #  Compute catch22 Features
    ###########################################################################
    # Helper to compute catch22 for a single 1D array
    def compute_catch22_for_one_series(ts_1d: np.ndarray,
                                       use_catch24: bool = False) -> dict:
        """
        Return {feature_name: feature_value} for the given 1D time series
        """
        ts_1d_list = ts_1d.tolist()
        results    = pycatch22.catch22_all(ts_1d_list, catch24=use_catch24)
        feat_dict  = {}
        for feat_name, feat_val in zip(results['names'], results['values']):
            feat_dict[feat_name] = feat_val
        return feat_dict

    # (D) For HUP: compute features for each electrode (column)
    hup_features_list = []
    for elec_name in hup_df.columns:
        single_ts  = hup_df[elec_name].values
        feat_dict  = compute_catch22_for_one_series(single_ts, use_catch24=False)
        feat_dict["electrode"] = elec_name
        hup_features_list.append(feat_dict)
    hup_features_df = pd.DataFrame(hup_features_list)
    print("\n[INFO] Computed catch22 features for HUP. Example rows:")
    print(hup_features_df.head())

    # (E) For MNI: similarly
    mni_features_list = []
    for elec_name in mni_df.columns:
        single_ts  = mni_df[elec_name].values
        feat_dict  = compute_catch22_for_one_series(single_ts, use_catch24=False)
        feat_dict["electrode"] = elec_name
        mni_features_list.append(feat_dict)
    mni_features_df = pd.DataFrame(mni_features_list)
    print("\n[INFO] Computed catch22 features for MNI. Example rows:")
    print(mni_features_df.head())

    # (Optional) Save results to CSV
    hup_features_df.to_csv("hup_catch22_features.csv", index=False)
    mni_features_df.to_csv("mni_catch22_features.csv", index=False)
    print("\n[DONE] Catch22 feature computation results saved to CSV.")


###############################################################################
# 6. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()