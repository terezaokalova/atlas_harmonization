#!/usr/bin/env python3
import os
import sys
import time
import pickle
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, hilbert, correlate, butter, filtfilt
from joblib import Parallel, delayed

# current_dir: /Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/src/terez_scripts/penn_all/multivar
current_dir = Path(__file__).resolve()
# Go up 4 levels to reach the root folder "atlas_harmonization"
project_root = current_dir.parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # from config.config import PENN_DATA_PATH
    from config.config import BASE_PATH
    print("Config imported successfully")
    print("BASE_PATH:", BASE_PATH)
    # print("Config imported successfully.")
    # print("PENN_DATA_PATH:", PENN_DATA_PATH)
except Exception as e:
    print("Error importing config:", e)
    sys.exit(1)

def segment_data(data, fs, win_size=2):
    win_samples = int(win_size * fs)
    n_windows = data.shape[0] // win_samples
    segments = [data[i*win_samples:(i+1)*win_samples, :] for i in range(n_windows)]
    return segments, n_windows

def pearson_segment(seg):
    return np.corrcoef(seg, rowvar=False)

def compute_pearson_sequential(data, fs, win_size=2, segments=None):
    if segments is None:
        segments, _ = segment_data(data, fs, win_size)
    corr_mats = [pearson_segment(seg) for seg in segments]
    return np.nanmean(np.array(corr_mats), axis=0)

def compute_squared_pearson(data, fs, win_size=2, segments=None):
    return compute_pearson_sequential(data, fs, win_size, segments)**2

def cross_corr_segment(seg):
    n_ch = seg.shape[1]
    cc = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            corr_full = correlate(seg[:, i], seg[:, j], mode='full', method='fft')
            norm = np.sqrt(np.sum(seg[:, i]**2) * np.sum(seg[:, j]**2))
            val = np.max(np.abs(corr_full)) / norm if norm != 0 else 0
            cc[i, j] = val
            cc[j, i] = val
    return cc

def compute_cross_correlation_sequential(data, fs, win_size=2, segments=None):
    if segments is None:
        segments, _ = segment_data(data, fs, win_size)
    cc_mats = [cross_corr_segment(seg) for seg in segments]
    return np.nanmean(np.array(cc_mats), axis=0)

def bp_filter(sig, fs, low, high):
    nyq = fs/2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

def plv_segment(seg, fs, low=8, high=12):
    n_ch = seg.shape[1]
    filtered = bp_filter(seg, fs, low, high)
    phase = np.angle(hilbert(filtered, axis=0))
    plv_mat = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            diff = phase[:, i] - phase[:, j]
            val = np.abs(np.sum(np.exp(1j*diff))) / len(diff)
            plv_mat[i, j] = val
            plv_mat[j, i] = val
    return plv_mat

def compute_plv_sequential(data, fs, win_size=2, low=8, high=12, segments=None):
    if segments is None:
        segments, _ = segment_data(data, fs, win_size)
    plv_mats = [plv_segment(seg, fs, low, high) for seg in segments]
    return np.nanmean(np.array(plv_mats), axis=0)

def re_segment(seg, fs, freqs):
    n_ch = seg.shape[1]
    n_freqs = freqs.shape[0]
    filtered_data = np.zeros((seg.shape[0], n_ch, n_freqs))
    for f in range(n_freqs):
        low, high = freqs[f]
        filtered_data[:, :, f] = bp_filter(seg, fs, low, high)
    re_seg = np.empty((n_ch, n_ch, n_freqs))
    for f in range(n_freqs):
        tmp_data = filtered_data[:, :, f]
        for i in range(n_ch):
            for j in range(i, n_ch):
                h1, _ = np.histogram(tmp_data[:, i], bins=10)
                h2, _ = np.histogram(tmp_data[:, j], bins=10)
                smooth = 1e-10
                h1 = h1 + smooth; h2 = h2 + smooth
                h1 /= np.sum(h1); h2 /= np.sum(h2)
                S1 = np.sum(h1 * np.log(h1/h2))
                S2 = np.sum(h2 * np.log(h2/h1))
                re_seg[i, j, f] = max(S1, S2)
                re_seg[j, i, f] = re_seg[i, j, f]
    return re_seg

def compute_relative_entropy_sequential(data, fs, win_size=2, freqs=np.array([[0.5,4],
                                                                              [4,8],
                                                                              [8,12],
                                                                              [12,30],
                                                                              [30,80]]), segments=None):
    if segments is None:
        segments, _ = segment_data(data, fs, win_size)
    re_mats = [re_segment(seg, fs, freqs) for seg in segments]
    avg = np.nanmean(np.array(re_mats), axis=0)
    if avg.ndim == 3 and avg.shape[2] == 1:
        return avg[:,:,0]
    return avg

def process_subject(subject_dir, win_size=2, freq_bands=np.array([[0.5,4],
                                                                 [4,8],
                                                                 [8,12],
                                                                 [12,30],
                                                                 [30,80]]),
                    group_index=None, n_groups=10):
    print(f"Processing subject: {subject_dir.name}")
    start_time = time.time()
    bipolar_df, fs = load_ieeg_data(subject_dir)
    data = bipolar_df.values
    segments, n_windows = segment_data(data, fs, win_size)
    if group_index is not None:
        group_size = max(1, len(segments) // n_groups)
        start = group_index * group_size
        end = start + group_size if group_index < n_groups - 1 else len(segments)
        segments = segments[start:end]
        print(f"Processing group {group_index+1}/{n_groups}: segments {start} to {end-1}")
    p_corr = compute_pearson_sequential(data, fs, win_size, segments)
    sq_corr = p_corr**2
    cc = compute_cross_correlation_sequential(data, fs, win_size, segments)
    plv = compute_plv_sequential(data, fs, win_size, low=freq_bands[2,0], high=freq_bands[2,1], segments=segments)
    re_mat = compute_relative_entropy_sequential(data, fs, win_size, freqs=freq_bands, segments=segments)
    if re_mat.ndim == 3 and re_mat.shape[2] == 1:
        re_mat = re_mat[:,:,0]
    fc_results = {
        'pearson': p_corr,
        'squared_pearson': sq_corr,
        'cross_correlation': cc,
        'plv': plv,
        'relative_entropy': re_mat
    }
    for key, matrix in fc_results.items():
        out_file = subject_dir / f"{subject_dir.name}_fc_{key}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(matrix, f)
        print(f"Saved {key} matrix: {matrix.shape} to {out_file}")
    duration = time.time() - start_time
    print(f"Finished processing {subject_dir.name} in {duration:.2f} seconds")
    return fc_results

def process_all_subjects(base_dir, win_size=2, freq_bands=np.array([[0.5,4],
                                                                   [4,8],
                                                                   [8,12],
                                                                   [12,30],
                                                                   [30,80]]),
                         group_index=None, n_groups=10):
    base_dir = Path(base_dir)
    subject_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    summary = {}
    for subj in subject_dirs:
        try:
            fc_results = process_subject(subj, win_size=win_size, freq_bands=freq_bands,
                                         group_index=group_index, n_groups=n_groups)
            summary[subj.name] = {'channels': fc_results['pearson'].shape[0],
                                  'pearson_shape': fc_results['pearson'].shape}
        except Exception as e:
            print(f"Error processing {subj.name}: {e}")
    print("Summary:")
    for subj, info in summary.items():
        print(f"{subj}: {info}")
    return summary

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        subject_dir = Path(sys.argv[1])
        group_index = int(sys.argv[2])
        process_subject(subject_dir, group_index=group_index)
    else:
        from config import PENN_DATA_PATH
        process_all_subjects(PENN_DATA_PATH)
