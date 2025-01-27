#!/usr/bin/env python3
"""
fooof_pipeline_agg.py

This script:
1) Loads HUP and MNI data.
2) Removes:
   - Resected/SOZ electrodes (HUP).
   - Bad-outcome patients (HUP).
3) Applies the same filtering (HPF=1Hz, LPF=80Hz, notch=60Hz) as used in the univariate pipeline
   *before* computing FOOOF features.
4) Computes power spectral densities for each electrode and applies FOOOF.
5) Aggregates the computed electrode-level FOOOF features by region, similarly to the previous aggregator code (for basic feats).

Usage:
    python fooof_pipeline_agg.py

Notes:
changes from fooof_pipeline.py:
dk_atlas_df = pd.read_csv(DK_ATLAS_PATH): Loads the Desikan–Killiany atlas lookup so we can merge roiNum to roi names.
aggregate_features_by_region(...) and aggregate_and_average_by_region(...) calls**: 
Generate two levels of region-based data:
Electrode-level aggregated by (patient, region).
Average of those region means across patients.
Each aggregated result is saved to CSV for both HUP and MNI datasets.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch, iirnotch
from fooof import FOOOF
from typing import Dict, Tuple, Set, Optional

###############################################################################
# PATHS & CONSTANTS
###############################################################################
DATA_BASE_PATH = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data"
RESULTS_BASE = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/results"

HUP_ATLAS_PATH = os.path.join(DATA_BASE_PATH, "hup_atlas.mat")
HUP_DF_PATH = os.path.join(DATA_BASE_PATH, "hup_df.csv")
MNI_ATLAS_PATH = os.path.join(DATA_BASE_PATH, "mni_atlas.mat")
MNI_DF_PATH = os.path.join(DATA_BASE_PATH, "mni_df.csv")
META_DATA_PATH = os.path.join(DATA_BASE_PATH, "metaData.csv")
DK_ATLAS_PATH = os.path.join(DATA_BASE_PATH, "desikanKilliany.csv")

HUP_TS_COL = "wake_clip"
MNI_TS_COL = "Data_W"
HUP_PATIENT_COL = "patient_no"
MNI_PATIENT_COL = "Patient"

###############################################################################
# OUTCOME-FILTER HELPERS (HUP)
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
# SIGNAL PROCESSING HELPERS
###############################################################################
def filter_signal_1d(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply:
      1) Low-pass at 80 Hz
      2) High-pass at 1 Hz
      3) Notch at 60 Hz
    Return filtered 1D data.
    """
    b, a = butter(3, 80 / (fs / 2), btype='low')
    sig_filtered = filtfilt(b, a, sig.astype(float))

    b, a = butter(3, 1 / (fs / 2), btype='high')
    sig_filtered = filtfilt(b, a, sig_filtered)

    b, a = iirnotch(60, 30, fs)
    sig_filtered = filtfilt(b, a, sig_filtered)

    return sig_filtered

def compute_psd(data: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    """
    f, psd = welch(data, fs=fs, window='hamming', nperseg=fs * 2, noverlap=fs, scaling='density')
    # Remove 60Hz noise band
    noise_mask = (f >= 57.5) & (f <= 62.5)
    return f[~noise_mask], psd[~noise_mask]

###############################################################################
# ELECTRODE-FILTER HELPERS
###############################################################################
def get_good_electrodes_resected_soz(atlas: Dict) -> np.ndarray:
    """
    Identify 'good' electrodes that are not resected and not in SOZ.
    If resected_ch / soz_ch don't exist, returns all electrodes.
    """
    if not all(k in atlas for k in ['resected_ch', 'soz_ch']):
        print("Warning: 'resected_ch' or 'soz_ch' not found in atlas. Returning ALL electrodes.")
        # fallback: assume we want them all
        # (data is 'wake_clip' with shape (#samples, #electrodes))
        return np.arange(atlas['wake_clip'].shape[1]) if 'wake_clip' in atlas else np.array([])
    resected = atlas['resected_ch'].flatten().astype(bool)
    soz      = atlas['soz_ch'].flatten().astype(bool)
    good_mask = ~(resected | soz)
    return np.where(good_mask)[0]

###############################################################################
# REGION AGGREGATORS
###############################################################################
def aggregate_features_by_region(
    features_df: pd.DataFrame,
    region_df: pd.DataFrame,
    dk_atlas_df: pd.DataFrame,
    patient_map_arr: np.ndarray
) -> pd.DataFrame:
    """
    Aggregate FOOOF features by region, grouped by (patient_id, roiNum).
    """
    combined_df = pd.concat([
        features_df.reset_index(drop=True),
        pd.DataFrame({
            'roiNum': region_df['roiNum'].reset_index(drop=True),
            'patient_id': patient_map_arr
        })
    ], axis=1)

    feat_cols = [col for col in combined_df.columns if col not in ['electrode', 'roiNum', 'patient_id', 'roi']]
    grouped_list = []

    for (pat_id, roi), group in combined_df.groupby(['patient_id', 'roiNum']):
        row_dict = {'patient_id': pat_id, 'roiNum': roi}
        for feat in feat_cols:
            row_dict[f"{feat}_mean"] = group[feat].mean()
        grouped_list.append(row_dict)

    region_features_df = pd.DataFrame(grouped_list)

    if 'roiNum' in dk_atlas_df.columns and 'roi' in dk_atlas_df.columns:
        region_features_df = pd.merge(
            region_features_df,
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )

    return region_features_df


def aggregate_and_average_by_region(
    features_df: pd.DataFrame,
    region_df: pd.DataFrame,
    dk_atlas_df: pd.DataFrame,
    patient_map_arr: np.ndarray
) -> pd.DataFrame:
    """
    Average FOOOF features within regions for each patient, then across patients.
    """
    combined_df = pd.concat([
        features_df.reset_index(drop=True),
        pd.DataFrame({
            'roiNum': region_df['roiNum'].reset_index(drop=True),
            'patient_id': patient_map_arr
        })
    ], axis=1)

    patient_region_avg = combined_df.groupby(['patient_id', 'roiNum']).mean(numeric_only=True)
    region_avg = patient_region_avg.groupby('roiNum').mean(numeric_only=True).reset_index()

    if 'roiNum' in dk_atlas_df.columns and 'roi' in dk_atlas_df.columns:
        region_avg = pd.merge(
            region_avg,
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )

    return region_avg

###############################################################################
# FOOOF FEATURE EXTRACTION
###############################################################################
def compute_fooof_features(f: np.ndarray, psd: np.ndarray, freq_range: Tuple[int, int] = (1, 40)) -> Dict:
    """
    Fit FOOOF model and extract features.
    """
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode='knee')
    fm.fit(f, psd, freq_range)
    return {
        'aperiodic_offset': fm.aperiodic_params_[0],
        'aperiodic_exponent': fm.aperiodic_params_[1],
        'r_squared': fm.r_squared_,
        'error': fm.error_,
        'num_peaks': fm.n_peaks_
    }

###############################################################################
# MAIN PIPELINE
###############################################################################
def process_cohort(prefix: str, atlas_path: str, region_path: str, ts_col: str, patient_col: str, good_outcome_patients: Optional[Set[int]], filter_good_elec: bool, fs: int) -> pd.DataFrame:
    """
    Load and process cohort data, filter electrodes, and compute FOOOF features.
    """
    atlas_dict = sio.loadmat(atlas_path)
    region_df = pd.read_csv(region_path)

    ts_data = atlas_dict[ts_col]
    if ts_data.shape[0] < ts_data.shape[1]:
        ts_data = ts_data.T

    patient_nums = atlas_dict[patient_col].flatten()

    electrode_mask = np.arange(ts_data.shape[1])
    if prefix == 'hup' and filter_good_elec:
        electrode_mask = get_good_electrodes_resected_soz(atlas_dict)

    if good_outcome_patients is not None:
        good_idxs = filter_by_outcome_and_build_good_indices(patient_nums, good_outcome_patients)
        electrode_mask = np.intersect1d(electrode_mask, good_idxs)

    ts_data = ts_data[:, electrode_mask]
    region_df = region_df.iloc[electrode_mask].reset_index(drop=True)
    patient_nums = patient_nums[electrode_mask]

    features_list = []
    for col_idx in range(ts_data.shape[1]):
        signal = ts_data[:, col_idx]
        filtered_signal = filter_signal_1d(signal, fs)
        f, psd = compute_psd(filtered_signal, fs)
        features = compute_fooof_features(f, psd)
        features['electrode'] = f"{prefix}_elec_{col_idx}"
        features_list.append(features)

    return pd.DataFrame(features_list), region_df, patient_nums

###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    # Load outcome info for HUP
    outcomes_dict, good_outcomes_set = load_and_process_outcomes(META_DATA_PATH, HUP_ATLAS_PATH)

    # Process HUP
    hup_features_df, hup_region_df, hup_patients_kept = process_cohort(
        prefix='hup',
        atlas_path=HUP_ATLAS_PATH,
        region_path=HUP_DF_PATH,
        ts_col=HUP_TS_COL,
        patient_col=HUP_PATIENT_COL,
        good_outcome_patients=good_outcomes_set,
        filter_good_elec=True,
        fs=200
    )

    # Process MNI
    mni_features_df, mni_region_df, mni_patients_kept = process_cohort(
        prefix='mni',
        atlas_path=MNI_ATLAS_PATH,
        region_path=MNI_DF_PATH,
        ts_col=MNI_TS_COL,
        patient_col=MNI_PATIENT_COL,
        good_outcome_patients=None,
        filter_good_elec=False,
        fs=200
    )

    # Save electrode-level features
    hup_features_df.to_csv(os.path.join(RESULTS_BASE, "hup_fooof_features.csv"), index=False)
    mni_features_df.to_csv(os.path.join(RESULTS_BASE, "mni_fooof_features.csv"), index=False)

    # -------------------------------------------------------------------------
    # AGGREGATE & SAVE REGION-LEVEL FEATURES
    # -------------------------------------------------------------------------
    dk_atlas_df = pd.read_csv(DK_ATLAS_PATH)

    # Aggregated by region (per patient)
    hup_region_features = aggregate_features_by_region(
        hup_features_df,
        hup_region_df,
        dk_atlas_df,
        hup_patients_kept
    )
    mni_region_features = aggregate_features_by_region(
        mni_features_df,
        mni_region_df,
        dk_atlas_df,
        mni_patients_kept
    )

    hup_region_features.to_csv(os.path.join(RESULTS_BASE, "hup_fooof_features_region.csv"), index=False)
    mni_region_features.to_csv(os.path.join(RESULTS_BASE, "mni_fooof_features_region.csv"), index=False)

    # Aggregated then averaged across patients
    hup_region_avg = aggregate_and_average_by_region(
        hup_features_df,
        hup_region_df,
        dk_atlas_df,
        hup_patients_kept
    )
    mni_region_avg = aggregate_and_average_by_region(
        mni_features_df,
        mni_region_df,
        dk_atlas_df,
        mni_patients_kept
    )

    hup_region_avg.to_csv(os.path.join(RESULTS_BASE, "hup_fooof_features_region_avg.csv"), index=False)
    mni_region_avg.to_csv(os.path.join(RESULTS_BASE, "mni_fooof_features_region_avg.csv"), index=False)
    # -------------------------------------------------------------------------

    print("[DONE] FOOOF feature extraction completed.")
