#!/usr/bin/env python3
"""
Full feature extraction pipeline that computes:
- PSD-based bandpower features,
- FOOOF parameters,
- Shannon entropy (time-domain),
- catch22 features (time-domain),
for each electrode (row) in each epoch file.

It saves the resulting table as both a pickle and a CSV file,
preserving electrode labels as the DataFrame index (if present).
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, iirnotch
import logging
from dataclasses import dataclass
import pycatch22  # make sure this is installed!
from fooof import FOOOF
from clean_hup_data_loading import get_clean_hup_file_paths, load_epoch
import sys

# Set up logging for information during processing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Python executable:", sys.executable)
print("sys.path:", sys.path)


@dataclass
class SpectralConfig:
    bands: dict
    sampling_frequency: int
    window_samples: int = None  # in samples

    def __post_init__(self):
        # If no window length is provided, default to 4 seconds
        if self.window_samples is None:
            self.window_samples = self.sampling_frequency * 4


class FeatureExtractor:
    def __init__(self, config: dict):
        """
        Initialize the feature extractor.
        The config dict must have:
          - config['preprocessing']['sampling_frequency'] -> int
          - config['features']['spectral']['bands'] -> dict of band edges
        """
        self.config = config
        self.sampling_frequency = config['preprocessing']['sampling_frequency']
        self.spectral_config = SpectralConfig(
            bands=config['features']['spectral']['bands'],
            sampling_frequency=self.sampling_frequency
        )
    
    def extract_features_from_epoch(self, epoch_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each electrode in epoch_df (each row), compute features:
         - PSD-based bandpower,
         - FOOOF parameters,
         - Shannon entropy,
         - catch22 features.
        Then append these features to the original metadata.
        Returns a DataFrame with all original metadata plus new feature columns.
        """
        feature_rows = []
        for _, row in epoch_df.iterrows():
            # Get the raw 1D EEG signal for this electrode.
            data = row['data']

            # 1) Filter the data (in the time domain) once
            filtered_data = self._apply_filters(data)

            # 2) Compute the full PSD (do not mask frequencies here)
            f_full, pxx_full = self._compute_psd(filtered_data)

            # 3) Compute bandpower features:
            #    Remove the notch region from the PSD before integration.
            mask = (f_full < 57.5) | (f_full > 62.5)
            f_masked = f_full[mask]
            pxx_masked = pxx_full[mask]
            bandpower_feats = self._compute_bandpower_from_psd(f_masked, pxx_masked)

            # 4) Compute FOOOF features from the full (unmasked) PSD.
            fooof_feats = self._compute_fooof_from_psd(f_full, pxx_full)

            # 5) Compute Shannon entropy from the already filtered time-domain data.
            entropy_feats = self._compute_entropy(filtered_data, window_size_sec=5, stride_sec=5)

            # 6) Compute catch22 features from the raw (unfiltered) data.
            catch22_feats = self._compute_catch22(data)

            # Merge all computed features with the original metadata.
            new_feats = {}
            new_feats.update(bandpower_feats)
            new_feats.update(fooof_feats)
            new_feats.update(entropy_feats)
            new_feats.update(catch22_feats)
            combined = row.to_dict()
            combined.update(new_feats)
            feature_rows.append(combined)

        features_df = pd.DataFrame(feature_rows)

        # If a 'labels' column exists, use it as the index.
        if 'labels' in features_df.columns:
            features_df.set_index('labels', inplace=True)

        return features_df

    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply common filtering in the time domain:
          - Low-pass at 80 Hz,
          - High-pass at 1 Hz,
          - Notch at 60 Hz.
        """
        fs = self.sampling_frequency
        data = np.asarray(data, dtype=float).flatten()
        b, a = butter(3, 80/(fs/2), btype='low')
        data = filtfilt(b, a, data)
        b, a = butter(3, 1/(fs/2), btype='high')
        data = filtfilt(b, a, data)
        b, a = iirnotch(60, 30, fs)
        data = filtfilt(b, a, data)
        return data

    def _compute_psd(self, data: np.ndarray):
        """
        Compute the power spectral density (PSD) using Welch’s method.
        Returns the full frequency vector and PSD (without masking out the notch).
        """
        fs = self.sampling_frequency
        nperseg = self.spectral_config.window_samples
        noverlap = nperseg // 2
        window = np.hamming(nperseg)
        f, pxx = welch(data, fs=fs, window=window,
                       nperseg=nperseg, noverlap=noverlap,
                       scaling='density')
        return f, pxx

    def _compute_bandpower_from_psd(self, f: np.ndarray, pxx: np.ndarray) -> dict:
        """
        Compute bandpower features (absolute, relative, and log-transformed) for each frequency band.
        """
        band_powers = {}
        for band_name, (fmin, fmax) in self.spectral_config.bands.items():
            idx = (f >= fmin) & (f <= fmax)
            # Use np.trapezoid (the modern replacement for np.trapz)
            power = np.trapezoid(pxx[idx], f[idx])
            band_powers[band_name] = power

        total_power = sum(band_powers.values()) + 1e-8
        out = {}
        for band_name, pwr in band_powers.items():
            out[f'{band_name}_power'] = pwr
            out[f'{band_name}_rel'] = pwr / total_power
            out[f'{band_name}_log'] = np.log10(pwr + 1)
        return out

    # nans before
    def _compute_fooof_from_psd(self, f: np.ndarray, pxx: np.ndarray) -> dict:
        fm = FOOOF()  # use default parameters
        # Replace any zero or negative values in the PSD with a small constant
        pxx_safe = np.where(pxx <= 0, 1e-12, pxx)
        fm.fit(f, pxx_safe, [1, 80])  # Fit over 1-80 Hz
        feats = {
            'fooof_aperiodic_offset': fm.aperiodic_params_[0],
            'fooof_aperiodic_exponent': fm.aperiodic_params_[1],
            'fooof_r_squared': fm.r_squared_,
            'fooof_error': fm.error_,
            'fooof_num_peaks': fm.n_peaks_
        }
        return feats
    # def _compute_fooof_from_psd(self, f: np.ndarray, pxx: np.ndarray) -> dict:
    #     """
    #     Fit FOOOF (using default parameters) over 1-80 Hz on the full PSD.
    #     Returns a dictionary with FOOOF output.
    #     """
    #     fm = FOOOF()  # Using default FOOOF settings.
    #     fm.fit(f, pxx, [1, 80])  # Fit over 1-80 Hz.
    #     feats = {
    #         'fooof_aperiodic_offset': fm.aperiodic_params_[0],
    #         'fooof_aperiodic_exponent': fm.aperiodic_params_[1],
    #         'fooof_r_squared': fm.r_squared_,
    #         'fooof_error': fm.error_,
    #         'fooof_num_peaks': fm.n_peaks_
    #     }
    #     return feats

    def _compute_entropy(self, data: np.ndarray, window_size_sec=5, stride_sec=5) -> dict:
        """
        Compute Shannon entropy over windows of the filtered data.
        Returns log10(mean_entropy + 1) as the feature.
        """
        fs = self.sampling_frequency
        w_samp = int(window_size_sec * fs)
        s_samp = int(stride_sec * fs)
        n_samp = len(data)

        entropies = []
        nan_count = 0
        total_wins = 0
        start = 0
        while start + w_samp <= n_samp:
            total_wins += 1
            seg = data[start:start + w_samp]
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(seg == 0):
                nan_count += 1
                start += s_samp
                continue
            try:
                hist, _ = np.histogram(seg, bins='auto', density=True)
            except ValueError:
                nan_count += 1
                start += s_samp
                continue
            if hist.sum() == 0:
                nan_count += 1
                start += s_samp
                continue
            p = hist[hist > 0]
            p = p / p.sum()
            H = -np.sum(p * np.log2(p))
            entropies.append(H)
            start += s_samp

        logger.info(f"Entropy: processed {total_wins} windows; {nan_count} skipped due to invalid data.")
        mean_ent = np.mean(entropies) if entropies else 0
        return {'entropy_5secwin': np.log10(mean_ent + 1)}

    def _compute_catch22(self, data: np.ndarray) -> dict:
        """
        Compute catch22 features on the raw time series.
        Returns a dictionary with keys prefixed by 'catch22_'.
        """
        ts_list = data.tolist()
        res = pycatch22.catch22_all(ts_list, catch24=False)
        out = {}
        for nm, val in zip(res['names'], res['values']):
            out[f'catch22_{nm}'] = val
        return out


def run_feature_extraction(base_path, config):
    """
    Runs the feature extraction pipeline on the clean HUP dataset.
    For each subject and epoch:
      - Loads the epoch's metadata table (each row corresponds to one electrode).
      - Computes new features (bandpower, FOOOF, entropy, catch22) and appends them.
      - Saves the resulting DataFrame as both a pickle and a CSV (with electrode labels as the index).
    """
    file_paths_dict = get_clean_hup_file_paths(base_path)
    extractor = FeatureExtractor(config)

    for subject, epoch_dict in file_paths_dict.items():
        subject_dir = os.path.join(base_path, subject)
        for epoch_idx, file_path in epoch_dict.items():
            epoch_df = load_epoch(file_path)
            feat_df = extractor.extract_features_from_epoch(epoch_df)

            # Save as pickle (only if file does not already exist)
            pkl_name = f"metadata_and_features_epch{epoch_idx}.pkl"
            pkl_path = os.path.join(subject_dir, pkl_name)
            # if not os.path.exists(pkl_path):
            feat_df.to_pickle(pkl_path)
            logger.info(f"Saved features for {subject}, epoch {epoch_idx} to {pkl_path}")
            # else:
            #     logger.info(f"File {pkl_path} already exists; skipping.")

            # Save as CSV with index preserved (to keep original electrode labels if present)
            csv_name = f"metadata_and_features_epch{epoch_idx}.csv"
            csv_path = os.path.join(subject_dir, csv_name)
            feat_df.to_csv(csv_path, index=True)
            logger.info(f"Saved CSV (with electrode labels as index) for {subject}, epoch {epoch_idx} to {csv_path}")


if __name__ == '__main__':
    config = {
        'preprocessing': {
            'sampling_frequency': 200
        },
        'features': {
            'spectral': {
                'bands': {
                    'delta': [0.5, 4],
                    'theta': [4, 8],
                    'alpha': [8, 12],
                    'beta': [12, 30],
                    'gamma': [30, 80]
                }
            }
        }
    }

    base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
    run_feature_extraction(base_path, config)
