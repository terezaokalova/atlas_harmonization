# catch22_pipeline_agg.py

#!/usr/bin/env python3
"""
catch22_pipeline_agg.py

This script:
1) Loads HUP and MNI data.
2) removes:
   - Resected/SOZ electrodes (HUP).
   - Bad-outcome patients (HUP).
3) Applies the same filtering (HPF=1Hz, LPF=80Hz, notch=60Hz) used in the univariate pipeline
   *before* computing catch22 features (except we do NOT do window-based segmentation).
4) Aggregates the computed electrode-level features by region, similarly to the previous aggregator code.

Usage:
    python catch22_pipeline_agg.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, Tuple, Set, Optional
import pycatch22

# Filtering utilities
from scipy.signal import butter, filtfilt, iirnotch

###############################################################################
# 1. PATHS & CONSTANTS
###############################################################################
# Adjust these as necessary
DATA_BASE_PATH  = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data"
RESULTS_BASE    = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/results"

HUP_ATLAS_PATH  = os.path.join(DATA_BASE_PATH, "hup_atlas.mat")
HUP_DF_PATH     = os.path.join(DATA_BASE_PATH, "hup_df.csv")
MNI_ATLAS_PATH  = os.path.join(DATA_BASE_PATH, "mni_atlas.mat")
MNI_DF_PATH     = os.path.join(DATA_BASE_PATH, "mni_df.csv")
META_DATA_PATH  = os.path.join(DATA_BASE_PATH, "metaData.csv")
DK_ATLAS_PATH   = os.path.join(DATA_BASE_PATH, "desikanKilliany.csv")

# HUP vs. MNI time-series columns
HUP_TS_COL      = "wake_clip"
MNI_TS_COL      = "Data_W"
HUP_PATIENT_COL = "patient_no"
MNI_PATIENT_COL = "Patient"

###############################################################################
# 2. OUTCOME-FILTER HELPERS (HUP)
###############################################################################
def load_and_process_outcomes(
    meta_data_path: str,
    hup_atlas_path: str
) -> Tuple[Dict[int, bool], Set[int]]:
    """
    Load metadata and create a "good outcome" mapping for HUP.

    'Engel_12_mo' and 'Engel_24_mo' in the CSV
    determine "good" if max(Engel_12_mo, Engel_24_mo) < 1.3.
    """
    meta_df = pd.read_csv(meta_data_path)
    hup_dict = sio.loadmat(hup_atlas_path)
    if 'patient_no' not in hup_dict:
        raise KeyError("No 'patient_no' found in HUP atlas—can't map outcomes.")

    patient_nos_in_hup = np.unique(hup_dict['patient_no'].flatten())

    outcomes_dict = {}
    good_outcomes = set()

    for i, row in meta_df.iterrows():
        if i >= len(patient_nos_in_hup):
            break
        patient_no  = patient_nos_in_hup[i]
        max_engel   = max(row['Engel_12_mo'], row['Engel_24_mo'])
        is_good     = (max_engel < 1.3)
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
    Return the indices of electrodes that belong to good-outcome patients.
    """
    good_idx_list = [
        i for i, pat_no in enumerate(patient_nums)
        if pat_no in good_outcome_patients
    ]
    return np.array(good_idx_list, dtype=int)

###############################################################################
# 3. ELECTRODE-FILTER HELPERS
###############################################################################
def get_good_electrodes_resected_soz(atlas: Dict) -> np.ndarray:
    """
    Identify 'good' electrodes that are not resected and not in SOZ.
    If resected_ch / soz_ch don't exist, returns all electrodes.
    """
    if not all(k in atlas for k in ['resected_ch', 'soz_ch']):
        print("Warning: 'resected_ch' or 'soz_ch' not found in atlas. Returning ALL electrodes.")
        # fallback: assume we want them all
        # (data is typically 'wake_clip' with shape (#samples, #electrodes), handle carefully)
        return np.arange(atlas['wake_clip'].shape[1]) if 'wake_clip' in atlas else np.array([])
    resected = atlas['resected_ch'].flatten().astype(bool)
    soz      = atlas['soz_ch'].flatten().astype(bool)
    good_mask = ~(resected | soz)
    return np.where(good_mask)[0]

###############################################################################
# 4. REGION AGGREGATORS
###############################################################################
def aggregate_features_by_region(
    features_df: pd.DataFrame,
    region_df: pd.DataFrame,
    dk_atlas_df: pd.DataFrame,
    patient_map_arr: np.ndarray
) -> pd.DataFrame:
    """
    Similar to univariate aggregator (for PSD+entropy feats, modular OOP version)). 
    - Takes 'features_df' (one row per electrode) and merges region info + patient ID.
    - Then groups by (patient_id, roiNum), computing means for each feature.
    - Merges region name from dk_atlas_df to get 'roi'.

    Assumes 'region_df' has a 'roiNum' column for each electrode
    in the same order as final DataFrame rows.
    """
    # combined: attach roiNum & patient_id to each row
    combined_df = pd.concat([
        features_df.reset_index(drop=True),
        pd.DataFrame({
            'roiNum': region_df['roiNum'].reset_index(drop=True),
            'patient_id': patient_map_arr
        })
    ], axis=1)

    # Get numeric (non-string) columns for grouping, ignoring 'electrode'
    # or other columns you don't want to average
    feat_cols = [col for col in combined_df.columns
                 if col not in ['electrode', 'roiNum', 'patient_id', 'roi']]

    grouped_list = []
    for (pat_id, roi), group in combined_df.groupby(['patient_id', 'roiNum']):
        row_dict = {
            'patient_id': pat_id,
            'roiNum': roi
        }
        for feat in feat_cols:
            row_dict[f"{feat}_mean"] = group[feat].mean()
        grouped_list.append(row_dict)

    region_features_df = pd.DataFrame(grouped_list)

    # add region name from dk_atlas
    if 'roiNum' in dk_atlas_df.columns and 'roi' in dk_atlas_df.columns:
        region_features_df = pd.merge(
            region_features_df,
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )
    else:
        print("Warning: No 'roiNum'/'roi' columns found in dk_atlas_df. Skipping region name merge.")

    return region_features_df


def aggregate_and_average_by_region(
    features_df: pd.DataFrame,
    region_df: pd.DataFrame,
    dk_atlas_df: pd.DataFrame,
    patient_map_arr: np.ndarray
) -> pd.DataFrame:
    """
    - Step 1: Combine electrode-level features with region + patient info
    - Step 2: Average within each (patient, roiNum)
    - Step 3: Then average across all patients for each region.
    """
    combined_df = pd.concat([
        features_df.reset_index(drop=True),
        pd.DataFrame({
            'roiNum': region_df['roiNum'].reset_index(drop=True),
            'patient_id': patient_map_arr
        })
    ], axis=1)

    # Group 1: average within patient-region
    patient_region_avg = combined_df.groupby(['patient_id', 'roiNum']).mean(numeric_only=True)

    # Group 2: average across patients
    region_avg = patient_region_avg.groupby('roiNum').mean(numeric_only=True).reset_index()

    # Merge roi name
    if 'roiNum' in dk_atlas_df.columns and 'roi' in dk_atlas_df.columns:
        region_avg = pd.merge(
            region_avg,
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )
    return region_avg

###############################################################################
# 5. LOADING & PREPROCESSING A COHORT
###############################################################################
def load_and_preprocess_cohort(
    prefix: str,
    atlas_path: str,
    region_path: str,
    ts_col: str,
    patient_col: str,
    good_outcome_patients: Optional[Set[int]] = None,
    filter_good_elec: bool = True
# ) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
# the correct way to specify a function returning multiple types in a specific order
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    1) Load .mat + region .csv
    2) Possibly filter resected/SOZ electrodes (hup only).
    3) Possibly filter out bad-outcome electrodes (if good_outcome_patients is given).
    4) Return:
       - A DataFrame (rows=samples, cols=kept electrodes),
       - Region df (subset to the same electrodes),
       - The final (electrode -> patient) map array (same shape as # electrodes).
    """
    print(f"\n--- Loading {prefix.upper()} data ---")
    atlas_dict = sio.loadmat(atlas_path)
    region_df  = pd.read_csv(region_path)

    if ts_col not in atlas_dict:
        raise KeyError(f"Column '{ts_col}' not found in {prefix} atlas data.")
    ts_data = atlas_dict[ts_col]

    print(f"  Original shape from '{ts_col}' is {ts_data.shape}")
    # check if transpose is needed
    need_transpose = (ts_data.shape[0] < ts_data.shape[1])
    print(f"  (Check) #rows < #cols => {need_transpose}")
    if need_transpose:
        ts_data = ts_data.T
        print(f"  Transposed shape => {ts_data.shape}")

    if patient_col not in atlas_dict:
        raise KeyError(f"Patient column '{patient_col}' not found in {prefix} atlas data.")
    all_patient_nums = atlas_dict[patient_col].flatten()

    # by default, keep all electrodes:
    electrode_mask = np.arange(ts_data.shape[1])

    # 1) Filter resected/SOZ if prefix='hup' and we want to:
    if prefix == 'hup' and filter_good_elec:
        good_idxs = get_good_electrodes_resected_soz(atlas_dict)
        print(f"  -> filtering resected/SOZ => keep {len(good_idxs)} of {ts_data.shape[1]}")
        electrode_mask = np.intersect1d(electrode_mask, good_idxs)

    # 2) Filter bad-outcome:
    if good_outcome_patients is not None:
        if len(all_patient_nums) != ts_data.shape[1]:
            raise ValueError("Mismatch in # of electrodes vs # of patient_no!")
        good_idxs = filter_by_outcome_and_build_good_indices(all_patient_nums, good_outcome_patients)
        print(f"  -> filtering out bad-outcome => keep {len(good_idxs)} of {ts_data.shape[1]}")
        electrode_mask = np.intersect1d(electrode_mask, good_idxs)

    # final subset
    ts_data = ts_data[:, electrode_mask]
    kept_patient_nums = all_patient_nums[electrode_mask]
    # region_df also needs to be subset in the same order:
    #   assume region_df is in the same original electrode order
    #   do the same indexing:
    region_df = region_df.iloc[electrode_mask].reset_index(drop=True)

    # build final DataFrame
    colnames = [f"{prefix}_elec_{i}" for i in range(ts_data.shape[1])]
    df_ts = pd.DataFrame(ts_data, columns=colnames)
    print(f"  Final shape after filtering => {df_ts.shape}")

    return df_ts, region_df, kept_patient_nums

###############################################################################
# 6. FILTERING HELPER FOR EEG
###############################################################################
def filter_signal_1d(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply:
      1) Low-pass at 80 Hz
      2) High-pass at 1 Hz
      3) Notch at 60 Hz
    Return filtered 1D data.
    """
    # lowpass at 80
    b, a = butter(3, 80 / (fs / 2), btype='low')
    sig_filtered = filtfilt(b, a, sig.astype(float))

    # highpass at 1
    b, a = butter(3, 1 / (fs / 2), btype='high')
    sig_filtered = filtfilt(b, a, sig_filtered)

    # notch at 60
    b, a = iirnotch(60, 30, fs)
    sig_filtered = filtfilt(b, a, sig_filtered)

    return sig_filtered

###############################################################################
# 7. MAIN: LOAD + PROCESS + FILTER + CATCH22 + AGGREGATION
###############################################################################
def main():
    """
    1) Load outcome info for HUP => get "good" set
    2) Load & preprocess HUP => filter electrodes/patients
    3) Load & preprocess MNI => skip outcome
    4) For each electrode => apply the HPF/LPF/notch filter => compute catch22
    5) Aggregate region-level (and region-average) if desired
    """

    ###########################################################################
    # A) LOAD OUTCOME (HUP)
    ###########################################################################
    outcomes_dict, good_outcomes_set = load_and_process_outcomes(META_DATA_PATH, HUP_ATLAS_PATH)

    ###########################################################################
    # B) LOAD + PREPROCESS HUP
    ###########################################################################
    hup_ts, hup_region_df, hup_patients_kept = load_and_preprocess_cohort(
        prefix='hup',
        atlas_path=HUP_ATLAS_PATH,
        region_path=HUP_DF_PATH,
        ts_col=HUP_TS_COL,
        patient_col=HUP_PATIENT_COL,
        good_outcome_patients=good_outcomes_set,  # filter out bad-outcome
        filter_good_elec=True                     # remove resected/SOZ
    )

    # we can attempt to read sampling frequency from atlas:
    # (if not found, default to 200 or whatever you used)
    atlas_dict = sio.loadmat(HUP_ATLAS_PATH)
    if 'SamplingFrequency' in atlas_dict:
        sfreq_hup = int(np.nan_to_num(atlas_dict['SamplingFrequency'].flatten())[0])
    else:
        sfreq_hup = 200
    print(f"[HUP] sampling freq = {sfreq_hup} Hz")

    ###########################################################################
    # C) LOAD + PREPROCESS MNI
    ###########################################################################
    mni_ts, mni_region_df, mni_patients_kept = load_and_preprocess_cohort(
        prefix='mni',
        atlas_path=MNI_ATLAS_PATH,
        region_path=MNI_DF_PATH,
        ts_col=MNI_TS_COL,
        patient_col=MNI_PATIENT_COL,
        good_outcome_patients=None,  # no outcome filter for MNI
        filter_good_elec=False       # skip resected/SOZ
    )

    atlas_dict_mni = sio.loadmat(MNI_ATLAS_PATH)
    if 'SamplingFrequency' in atlas_dict_mni:
        sfreq_mni = int(np.nan_to_num(atlas_dict_mni['SamplingFrequency'].flatten())[0])
    else:
        sfreq_mni = 200
    print(f"[MNI] sampling freq = {sfreq_mni} Hz")

    ###########################################################################
    # D) PREPARE CATCH22
    ###########################################################################
    def compute_catch22_for_one_series(ts_1d: np.ndarray,
                                       use_catch24: bool = False) -> dict:
        """
        1) Filter 1D data with HPF=1, LPF=80, Notch=60
        2) Compute catch22 on the filtered data
        """
        # apply your EEG-style filters
        filtered_ts = filter_signal_1d(ts_1d, fs=sfreq_hup if not use_catch24 else sfreq_hup)
        # (the same freq is used for MNI if we call function for MNI data, we might pass an argument for fs).
        # but let's keep it simpler: we create a separate function for MNI or pass an fs param. 
        # For demonstration, we do a single function. We'll override fs inside the loop if needed.

        results = pycatch22.catch22_all(filtered_ts.tolist(), catch24=use_catch24)
        feat_dict = {}
        for feat_name, feat_val in zip(results['names'], results['values']):
            feat_dict[feat_name] = feat_val
        return feat_dict

    ###########################################################################
    # E) Compute Catch22 for HUP (electrode-level)
    ###########################################################################
    hup_features_list = []
    for col in hup_ts.columns:
        # We re-filter each electrode's signal, but let's pass the correct sampling freq
        raw_1d = hup_ts[col].values
        # We'll do a version of compute_catch22_for_one_series that takes fs as param:
        filtered_ts = filter_signal_1d(raw_1d, fs=sfreq_hup)
        res = pycatch22.catch22_all(filtered_ts.tolist(), catch24=False)
        feat_dict = dict(zip(res['names'], res['values']))
        feat_dict["electrode"] = col
        hup_features_list.append(feat_dict)

    hup_features_df = pd.DataFrame(hup_features_list)
    print(f"\n[INFO] HUP catch22 done -> shape={hup_features_df.shape}")

    ###########################################################################
    # F) Compute Catch22 for MNI (electrode-level)
    ###########################################################################
    mni_features_list = []
    for col in mni_ts.columns:
        raw_1d = mni_ts[col].values
        filtered_ts = filter_signal_1d(raw_1d, fs=sfreq_mni)
        res = pycatch22.catch22_all(filtered_ts.tolist(), catch24=False)
        feat_dict = dict(zip(res['names'], res['values']))
        feat_dict["electrode"] = col
        mni_features_list.append(feat_dict)

    mni_features_df = pd.DataFrame(mni_features_list)
    print(f"[INFO] MNI catch22 done -> shape={mni_features_df.shape}")

    ###########################################################################
    # G) AGGREGATION BY REGION
    ###########################################################################
    # Need the same shape for region_df as in 'ts' => so we can line up electrode -> region
    # We also have 'kept_patient_nums' for each electrode -> patient ID
    dk_atlas_df = pd.read_csv(DK_ATLAS_PATH)

    # HUP region-level
    hup_region_features = aggregate_features_by_region(
        features_df   = hup_features_df,
        region_df     = hup_region_df,
        dk_atlas_df   = dk_atlas_df,
        patient_map_arr = hup_patients_kept
    )
    print(f"[INFO] Aggregated HUP region feats => {hup_region_features.shape}")

    # MNI region-level
    mni_region_features = aggregate_features_by_region(
        features_df     = mni_features_df,
        region_df       = mni_region_df,
        dk_atlas_df     = dk_atlas_df,
        patient_map_arr = mni_patients_kept
    )
    print(f"[INFO] Aggregated MNI region feats => {mni_region_features.shape}")

    # (Optionally) region-averages across patients:
    hup_region_avg = aggregate_and_average_by_region(
        features_df     = hup_features_df,
        region_df       = hup_region_df,
        dk_atlas_df     = dk_atlas_df,
        patient_map_arr = hup_patients_kept
    )
    mni_region_avg = aggregate_and_average_by_region(
        features_df     = mni_features_df,
        region_df       = mni_region_df,
        dk_atlas_df     = dk_atlas_df,
        patient_map_arr = mni_patients_kept
    )

    ###########################################################################
    # H) SAVE RESULTS
    ###########################################################################
    # 1) Electrode-level catch22
    hup_features_df.to_csv(os.path.join(RESULTS_BASE, "hup_catch22_feats_elec.csv"), index=False)
    mni_features_df.to_csv(os.path.join(RESULTS_BASE, "mni_catch22_feats_elec.csv"), index=False)

    # 2) Region-level
    hup_region_features.to_csv(os.path.join(RESULTS_BASE, "hup_catch22_feats_region.csv"), index=False)
    mni_region_features.to_csv(os.path.join(RESULTS_BASE, "mni_catch22_feats_region.csv"), index=False)

    # 3) Region-averages
    hup_region_avg.to_csv(os.path.join(RESULTS_BASE, "hup_catch22_feats_region_avg.csv"), index=False)
    mni_region_avg.to_csv(os.path.join(RESULTS_BASE, "mni_catch22_feats_region_avg.csv"), index=False)

    print("\n[DONE] catch22 pipeline with region aggregation complete.\n")


###############################################################################
# 8. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
