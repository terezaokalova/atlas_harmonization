#!/usr/bin/env python3
import os
import sys
import time
import pickle
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from joblib import Parallel, delayed

current_dir = Path(__file__).resolve()
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

try:
    from config.config import PENN_DATA_PATH
except ImportError:
    sys.exit(1)

# --------------------------
# Data Loading & Segmentation
# --------------------------
def load_ieeg_data(subject_path):
    h5_files = list(subject_path.rglob("interictal_ieeg_processed.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 file found in {subject_path}")
    with h5py.File(h5_files[0], 'r') as f:
        ieeg_data = f['/bipolar_montage/ieeg']
        bipolar_df = pd.DataFrame(ieeg_data, columns=ieeg_data.attrs['channels_labels'])
        fs = ieeg_data.attrs['sampling_rate']
    return bipolar_df, fs

def segment_data(data, fs, win_size=2):
    win_samples = int(win_size * fs)
    n_windows = data.shape[0] // win_samples
    segments = [data[i*win_samples:(i+1)*win_samples, :] for i in range(n_windows)]
    return segments, n_windows

def bp_filter(sig, fs, low, high):
    nyq = fs/2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

# --------------------------
# Optimized Cross-Correlation
# --------------------------
def cross_corr_segment(seg):
    n = seg.shape[0]
    n_ch = seg.shape[1]
    fft_all = np.fft.fft(seg, n=2*n, axis=0)
    cc = np.fft.ifft(fft_all[:, :, None] * fft_all.conj()[:, None, :], axis=0)
    cc = np.abs(cc[:n, :, :])
    max_cc = np.max(cc, axis=0)
    norms = np.sqrt(np.sum(seg**2, axis=0))
    norm_matrix = norms[:, None] * norms[None, :]
    return max_cc / (norm_matrix + 1e-10)

# --------------------------
# Optimized Relative Entropy
# --------------------------
def re_segment(seg, fs, freqs):
    n_ch = seg.shape[1]
    n_freqs = freqs.shape[0]
    filtered = np.stack([bp_filter(seg, fs, low, high) for (low, high) in freqs], axis=-1)
    bins = np.linspace(-1, 1, 11)
    # Compute histograms along time axis for each channel and frequency band.
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0],
                                 axis=0, arr=filtered)
    hists = (hists + 1e-10) / (hists.sum(axis=1, keepdims=True) + 1e-10)
    re_matrix = np.empty((n_ch, n_ch, n_freqs))
    for f in range(n_freqs):
        h = hists[:, :, f]
        S = np.maximum(np.sum(h[:, None] * np.log(h[:, None] / h[None, :]), axis=-1),
                       np.sum(h[None, :] * np.log(h[None, :] / h[:, None]), axis=-1))
        re_matrix[..., f] = S
    return re_matrix

# --------------------------
# Helper for Parallel Processing
# --------------------------
def parallel_compute(func, data, fs, win_size, **kwargs):
    segments, _ = segment_data(data, fs, win_size)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(func)(seg, fs, **kwargs) for seg in segments
    )
    return np.nanmean(np.array(results), axis=0)

# --------------------------
# Modified Processing Functions
# --------------------------
def compute_pearson(data, fs, win_size=2):
    return parallel_compute(lambda seg, _: np.corrcoef(seg.T), data, fs, win_size)

def compute_cross_correlation(data, fs, win_size=2):
    return parallel_compute(cross_corr_segment, data, fs, win_size)

def compute_plv(data, fs, win_size=2, low=8, high=12):
    def plv_task(seg, fs_, l, h):
        phase = np.angle(hilbert(bp_filter(seg, fs_, l, h), axis=0))
        comp = np.exp(1j * phase)
        return np.abs(comp.T @ comp.conj()) / phase.shape[0]
    return parallel_compute(plv_task, data, fs, win_size, l=low, h=high)

def compute_relative_entropy(data, fs, win_size=2, freqs=None):
    freqs = freqs or np.array([[0.5,4], [4,8], [8,12], [12,30], [30,80]])
    return parallel_compute(re_segment, data, fs, win_size, freqs=freqs)

# --------------------------
# Updated Process Subject/All Subjects
# --------------------------
def process_subject(subject_dir, win_size=2, freq_bands=None):
    print(f"Processing subject: {subject_dir.name}")
    start_time = time.time()
    bipolar_df, fs = load_ieeg_data(subject_dir)
    data = bipolar_df.values
    fc_results = {
        'pearson': compute_pearson(data, fs, win_size),
        'squared_pearson': compute_pearson(data, fs, win_size)**2,
        'cross_correlation': compute_cross_correlation(data, fs, win_size),
        'plv': compute_plv(data, fs, win_size, 8, 12),
        'relative_entropy': compute_relative_entropy(data, fs, win_size, freq_bands)
    }
    for key, mat in fc_results.items():
        out_file = subject_dir / f"{subject_dir.name}_fc_{key}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(mat, f)
        print(f"Saved {key} matrix: {mat.shape} to {out_file}")
    duration = time.time() - start_time
    print(f"Finished processing {subject_dir.name} in {duration:.2f} seconds")
    return fc_results

def process_all_subjects(base_dir, win_size=2, freq_bands=None, group_index=None, n_groups=10):
    base_dir = Path(base_dir)
    subject_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    summary = {}
    for subj in subject_dirs:
        try:
            fc_results = process_subject(subj, win_size=win_size, freq_bands=freq_bands)
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
        process_all_subjects(PENN_DATA_PATH, freq_bands=np.array([[0.5,4],
                                                                  [4,8],
                                                                  [8,12],
                                                                  [12,30],
                                                                  [30,80]]))
