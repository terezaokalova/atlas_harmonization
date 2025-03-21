#!/usr/bin/env python3
"""
Feature extraction pipeline for multivariate connectivity features.
Computes functional connectivity matrices using different metrics:
- Pearson correlation
- Squared Pearson correlation
- Cross-correlation
- Phase Locking Value (PLV)
"""

import sys
print("Running interpreter:", sys.executable)

import os
import sys
import time
import pickle
import logging
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("features_multivar")

# --------------------------
# Data Loading & Segmentation
# --------------------------
def load_ieeg_data(subject_path):
    """Load iEEG data from an H5 file in the subject directory."""
    h5_files = list(Path(subject_path).rglob("interictal_ieeg_processed.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 file found in {subject_path}")
    
    logger.info(f"Loading data from: {h5_files[0]}")
    try:
        with h5py.File(h5_files[0], 'r') as f:
            ieeg_data = f['/bipolar_montage/ieeg']
            bipolar_df = pd.DataFrame(ieeg_data[:], columns=ieeg_data.attrs['channels_labels'])
            fs = ieeg_data.attrs['sampling_rate']
        
        logger.info(f"Data loaded successfully: {bipolar_df.shape}, fs={fs}")
        return bipolar_df, fs
    except Exception as e:
        logger.error(f"Error loading H5 file: {e}")
        raise

def segment_data(data, fs, win_size=2):
    """Segment data into windows of specified size."""
    win_samples = int(win_size * fs)
    n_windows = data.shape[0] // win_samples
    segments = [data[i*win_samples:(i+1)*win_samples, :] for i in range(n_windows)]
    return segments, n_windows

def bp_filter(sig, fs, low, high):
    """Apply bandpass filter to signal."""
    nyq = fs/2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

# --------------------------
# Cross-Correlation
# --------------------------
def cross_corr_segment(seg, fs):
    """Compute cross-correlation for a segment of data."""
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
# Parallel Processing Helper
# --------------------------
def parallel_compute(func, data, fs, win_size, **kwargs):
    """Compute features in parallel across segments."""
    segments, _ = segment_data(data, fs, win_size)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(func)(seg, fs, **kwargs) for seg in segments
    )
    return np.nanmean(np.array(results), axis=0)

# --------------------------
# Processing Functions
# --------------------------
def compute_pearson(data, fs, win_size=2):
    """Compute Pearson correlation across segments."""
    return parallel_compute(lambda seg, _: np.corrcoef(seg.T), data, fs, win_size)

def compute_cross_correlation(data, fs, win_size=2):
    """Compute cross-correlation across segments."""
    return parallel_compute(cross_corr_segment, data, fs, win_size)

def compute_plv(data, fs, win_size=2, low=8, high=12):
    """Compute phase locking value across segments."""
    def plv_task(seg, fs_, l, h):
        phase = np.angle(hilbert(bp_filter(seg, fs_, l, h), axis=0))
        comp = np.exp(1j * phase)
        return np.abs(np.dot(comp.conj().T, comp)) / phase.shape[0]
    return parallel_compute(plv_task, data, fs, win_size, l=low, h=high)

# --------------------------
# Main Feature Extraction Function
# --------------------------
def extract_multivar_features(
    subject_id, 
    data_root, 
    win_size=2, 
    save_results=True,
    output_dir=None
):
    """
    Extract multivariate features for a given subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (with or without 'sub-' prefix)
    data_root : str or Path
        Root directory containing subject folders
    win_size : float, default=2
        Window size in seconds for segmentation
    save_results : bool, default=True
        Whether to save results to disk
    output_dir : str or Path, optional
        Directory to save results. If None, saves in subject directory
    
    Returns:
    --------
    dict
        Dictionary of computed features
    """
    start_time = time.time()
    logger.info(f"Processing subject: {subject_id}")
    
    # Handle subject_id with or without "sub-" prefix
    data_root = Path(data_root)
    subject_dir_name = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    subject_dir = data_root / subject_dir_name
    
    if not subject_dir.exists():
        # Try without "sub-" prefix as fallback
        subject_dir = data_root / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_id} or {subject_dir_name}")
    
    logger.info(f"Using subject directory: {subject_dir}")
    
    # Load and process data
    bipolar_df, fs = load_ieeg_data(subject_dir)
    data = bipolar_df.values
    
    # Compute features
    logger.info("Computing features...")
    fc_results = {
        'pearson': compute_pearson(data, fs, win_size),
        'squared_pearson': compute_pearson(data, fs, win_size)**2,
        'cross_correlation': compute_cross_correlation(data, fs, win_size),
        'plv': compute_plv(data, fs, win_size, 8, 12),
    }
    logger.info("Feature computation complete")
    
    # Save results if requested
    if save_results:
        if output_dir is None:
            output_dir = subject_dir
        else:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # Use the base subject_id without "sub-" prefix for file naming
        base_subject_id = subject_id.replace("sub-", "") if subject_id.startswith("sub-") else subject_id
        
        for key, mat in fc_results.items():
            out_file = output_dir / f"{base_subject_id}_fc_{key}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(mat, f)
            logger.info(f"Saved {key} matrix: {mat.shape} to {out_file}")
    
    duration = time.time() - start_time
    logger.info(f"Finished processing {subject_id} in {duration:.2f} seconds")
    
    return fc_results

# --------------------------
# Output Verification Function
# --------------------------
def verify_outputs(subject_id, data_root=None):
    """
    Verify the output files for a subject and print information about them.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (with or without 'sub-' prefix)
    data_root : str or Path, optional
        Root directory containing subject folders
    """
    # Handle paths
    if data_root is None:
        data_root = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/Penn"
    
    data_root = Path(data_root)
    subject_dir_name = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    subject_dir = data_root / subject_dir_name
    
    if not subject_dir.exists():
        subject_dir = data_root / subject_id
        if not subject_dir.exists():
            print(f"Subject directory not found: {subject_id}")
            return
    
    # Check the base subject ID for filenames
    base_subject_id = subject_id.replace("sub-", "") if subject_id.startswith("sub-") else subject_id
    
    # Look for output files
    feature_types = ['pearson', 'squared_pearson', 'cross_correlation', 'plv']
    
    # Also load the original data to check channel count
    try:
        bipolar_df, fs = load_ieeg_data(subject_dir)
        print(f"\nOriginal data information:")
        print(f"Number of channels: {bipolar_df.shape[1]}")
        print(f"Number of time points: {bipolar_df.shape[0]}")
        print(f"Sampling rate: {fs} Hz")
        print(f"Duration: {bipolar_df.shape[0]/fs:.2f} seconds")
    except Exception as e:
        print(f"Error loading original data: {e}")
    
    print("\nOutput files verification:")
    for feature in feature_types:
        file_path = subject_dir / f"{base_subject_id}_fc_{feature}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                print(f"\n{feature.upper()} matrix:")
                print(f"  Shape: {data.shape}")
                print(f"  Size: {data.size} elements")
                print(f"  Data type: {data.dtype}")
                print(f"  Min value: {data.min():.6f}")
                print(f"  Max value: {data.max():.6f}")
                print(f"  Mean value: {data.mean():.6f}")
                
                # Check if the matrix is square as expected for FC matrices
                if data.shape[0] == data.shape[1]:
                    print(f"Square matrix ({data.shape[0]}×{data.shape[1]})")
                else:
                    print(f"Non-square matrix: {data.shape}")
                
                # Check diagonal for correlation matrices (should be 1.0 for pearson)
                if feature in ['pearson', 'squared_pearson']:
                    diag_mean = np.mean(np.diag(data))
                    if np.isclose(diag_mean, 1.0 if feature == 'pearson' else 1.0**2):
                        print(f"Diagonal values correct (mean: {diag_mean:.6f})")
                    else:
                        print(f"Unexpected diagonal values: {diag_mean:.6f}")
            except Exception as e:
                print(f"Error loading {feature} matrix: {e}")
        else:
            print(f"{feature} matrix file not found: {file_path}")

# --------------------------
# Command Line Interface
# --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract multivariate features for a subject")
    parser.add_argument("subject_id", help="Subject identifier (with or without 'sub-' prefix)")
    parser.add_argument("--data-root", default="/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/Penn",
                        help="Root directory containing subject folders")
    parser.add_argument("--win-size", type=float, default=2, help="Window size in seconds")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing results, skip computation")
    
    args = parser.parse_args()
    
    # verification-only option
    if args.verify_only:
        print(f"Verifying existing results for {args.subject_id}...")
        verify_outputs(args.subject_id, args.data_root)
        sys.exit(0)
    
    try:
        features = extract_multivar_features(
            subject_id=args.subject_id,
            data_root=args.data_root,
            win_size=args.win_size,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        print(f"Successfully extracted features for {args.subject_id}")
        
        # Run verification
        print("\n--- Verifying output files ---")
        verify_outputs(args.subject_id, args.data_root)
        
    except Exception as e:
        import traceback
        print(f"Error processing {args.subject_id}: {e}")
        traceback.print_exc()
        sys.exit(1)