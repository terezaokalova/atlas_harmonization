# missing fooof
# #!/usr/bin/env python3
# """
# Full feature extraction pipeline that computes PSD, entropy, and catch22 features
# for each electrode (row) in each epoch file. It then saves the resulting feature table
# both as a pickle and a CSV file.
# Requirements:
#     pip install pycatch22 scipy numpy pandas
# """

# import os
# import numpy as np
# import pandas as pd
# from scipy.signal import welch, butter, filtfilt, iirnotch
# from scipy.stats import entropy
# import logging
# from dataclasses import dataclass
# import pycatch22  # make sure this is installed!
# from clean_hup_data_loading import get_clean_hup_file_paths, load_epoch
# import sys

# # Set up logging for information during processing
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# print("Python executable:", sys.executable)
# print("sys.path:", sys.path)


# @dataclass
# class SpectralConfig:
#     bands: dict
#     sampling_frequency: int
#     window_samples: int = None  # in samples

#     def __post_init__(self):
#         # Use a 4-second window for Welch
#         if self.window_samples is None:
#             self.window_samples = self.sampling_frequency * 4


# class FeatureExtractor:
#     def __init__(self, config: dict):
#         """
#         Initialize the feature extractor.
#         Expects a config dictionary with keys:
#             'preprocessing': {'sampling_frequency': int},
#             'features': {'spectral': {'bands': dict}}
#         """
#         self.config = config
#         self.sampling_frequency = config['preprocessing']['sampling_frequency']
#         self.spectral_config = SpectralConfig(
#             bands=config['features']['spectral']['bands'],
#             sampling_frequency=self.sampling_frequency
#         )

#     def extract_features_from_epoch(self, epoch_df: pd.DataFrame) -> pd.DataFrame:
#         """
#         For the given epoch DataFrame (each row = one electrode), compute features for each electrode.
#         The returned DataFrame will have all original metadata columns and new feature columns appended.
#         """
#         feature_rows = []
#         for idx, row in epoch_df.iterrows():
#             data = row['data']  # 1D EEG signal for one electrode

#             # Compute PSD features
#             psd_features = self._compute_normalized_psd(data)
#             # Compute entropy features over 5-second windows
#             entropy_features = self._compute_entropy_fullts(data, window_size_sec=5, stride_sec=5)
#             # Compute catch22 time-domain features
#             catch22_features = self._compute_catch22_features(data)

#             # Merge all features into one dictionary
#             new_features = {**psd_features, **entropy_features, **catch22_features}
#             combined = row.to_dict()
#             combined.update(new_features)
#             feature_rows.append(combined)
#         return pd.DataFrame(feature_rows)

#     def _compute_normalized_psd(self, data: np.ndarray) -> dict:
#         fs = self.sampling_frequency
#         window_samples = self.spectral_config.window_samples  # 4-second window in samples
#         noverlap = window_samples // 2  # 50% overlap

#         window = np.hamming(window_samples)
#         f, pxx = welch(data, fs=fs, window=window, nperseg=window_samples, noverlap=noverlap, scaling='density')
#         # Exclude the notch region (57.5 to 62.5 Hz)
#         mask = (f < 57.5) | (f > 62.5)
#         f = f[mask]
#         pxx = pxx[mask]

#         band_powers = {}
#         for band_name, freq_range in self.spectral_config.bands.items():
#             idx = (f >= freq_range[0]) & (f <= freq_range[1])
#             power = np.trapezoid(pxx[idx], f[idx])
#             band_powers[band_name] = power

#         total_power = sum(band_powers.values()) + 1e-8
#         features = {}
#         for band_name, power in band_powers.items():
#             features[f'{band_name}_power'] = power
#             features[f'{band_name}_rel'] = power / total_power
#             features[f'{band_name}_log'] = np.log10(power + 1)
#         return features

#     def _compute_entropy_fullts(self, data: np.ndarray, window_size_sec: float, stride_sec: float) -> dict:
#         fs = self.sampling_frequency
#         window_samples = int(window_size_sec * fs)
#         stride_samples = int(stride_sec * fs)
#         n_samples = len(data)

#         entropies = []
#         nan_count = 0
#         total_windows = 0

#         start = 0
#         while start + window_samples <= n_samples:
#             total_windows += 1
#             segment = data[start: start + window_samples]
#             segment_filt = self._apply_filters(segment)
#             segment_filt = np.nan_to_num(segment_filt, nan=0.0, posinf=0.0, neginf=0.0)

#             if np.all(segment_filt == 0):
#                 nan_count += 1
#                 start += stride_samples
#                 continue

#             try:
#                 hist, _ = np.histogram(segment_filt, bins='auto', density=True)
#             except ValueError as e:
#                 nan_count += 1
#                 start += stride_samples
#                 continue

#             if hist.sum() == 0:
#                 nan_count += 1
#                 start += stride_samples
#                 continue

#             p = hist[hist > 0]
#             p = p / p.sum()
#             H = -np.sum(p * np.log2(p))
#             entropies.append(H)
#             start += stride_samples

#         print(f"Entropy: Processed {total_windows} windows; {nan_count} windows skipped due to invalid data.")
#         mean_entropy = np.mean(entropies) if entropies else 0
#         entropy_feature = np.log10(mean_entropy + 1)
#         return {'entropy_5secwin': entropy_feature}

#     def _apply_filters(self, data: np.ndarray) -> np.ndarray:
#         fs = self.sampling_frequency
#         data = np.asarray(data).flatten()

#         b, a = butter(3, 80/(fs/2), btype='low')
#         data = filtfilt(b, a, data)
#         b, a = butter(3, 1/(fs/2), btype='high')
#         data = filtfilt(b, a, data)
#         b, a = iirnotch(60, 30, fs)
#         data = filtfilt(b, a, data)
#         return data

#     def _compute_catch22_features(self, data: np.ndarray) -> dict:
#         """
#         Compute catch22 features using pycatch22.
#         Returns a dictionary with keys prefixed by 'catch22_'.
#         """
#         ts_list = data.tolist()  # convert the time series to list
#         results = pycatch22.catch22_all(ts_list, catch24=False)
#         features = {}
#         for feat_name, feat_val in zip(results['names'], results['values']):
#             features[f'catch22_{feat_name}'] = feat_val
#         return features


# def run_feature_extraction(base_path, config):
#     """
#     Runs the feature extraction pipeline on the clean HUP dataset.
#     For each subject and each epoch:
#       - Loads the epoch's metadata table (each row corresponds to one electrode).
#       - Computes new features (PSD, entropy, catch22) for each electrode and appends them to the metadata.
#       - Saves the resulting DataFrame as a pickle file and also as a CSV (with electrode labels as a column)
#         in the same subject directory.
#     """
#     file_paths_dict = get_clean_hup_file_paths(base_path)
#     extractor = FeatureExtractor(config)

#     for subject, epochs_dict in file_paths_dict.items():
#         subject_dir = os.path.join(base_path, subject)
#         for epoch_idx, file_path in epochs_dict.items():
#             epoch_df = load_epoch(file_path)
#             features_df = extractor.extract_features_from_epoch(epoch_df)

#             # Save as pickle (only if the file does not already exist)
#             out_pkl_filename = f"metadata_and_features_epch{epoch_idx}.pkl"
#             out_pkl_path = os.path.join(subject_dir, out_pkl_filename)
#             # will want to reintroduce this eventually
#             if not os.path.exists(out_pkl_path):
#                 features_df.to_pickle(out_pkl_path)
#                 logger.info(f"Saved features for {subject}, epoch {epoch_idx} to {out_pkl_path}")
#             else:
#                 logger.info(f"File {out_pkl_path} already exists; skipping.")

#             # Save as CSV (preserve channel labels)
#             out_csv_filename = f"metadata_and_features_epch{epoch_idx}.csv"
#             out_csv_path = os.path.join(subject_dir, out_csv_filename)
#             features_df.to_csv(out_csv_path, index=True)
#             logger.info(f"Saved features (with original channel names as index) for {subject}, epoch {epoch_idx} to {out_csv_path}")
#             # out_csv_filename = f"metadata_and_features_epch{epoch_idx}.csv"
#             # out_csv_path = os.path.join(subject_dir, out_csv_filename)
#             # features_df_reset = features_df.reset_index()
#             # features_df_reset.to_csv(out_csv_path, index=False)
#             # logger.info(f"Saved features (with electrode labels) for {subject}, epoch {epoch_idx} to {out_csv_path}")

# if __name__ == '__main__':
#     config = {
#         'preprocessing': {
#             'sampling_frequency': 200
#         },
#         'features': {
#             'spectral': {
#                 'bands': {
#                     'delta': [0.5, 4],
#                     'theta': [4, 8],
#                     'alpha': [8, 12],
#                     'beta': [12, 30],
#                     'gamma': [30, 80]
#                 }
#             }
#         }
#     }

#     base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
#     run_feature_extraction(base_path, config)

# # import os
# # import numpy as np
# # import pandas as pd
# # from scipy.signal import welch, butter, filtfilt, iirnotch
# # from scipy.stats import entropy
# # import logging
# # from dataclasses import dataclass
# # from clean_hup_data_loading import get_clean_hup_file_paths, load_epoch
# # import sys

# # # Set up logging for information during processing
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # print("Python executable:", sys.executable)
# # print("sys.path:", sys.path)

# # @dataclass
# # class SpectralConfig:
# #     bands: dict
# #     sampling_frequency: int
# #     window_samples: int = None  # in samples

# #     def __post_init__(self):
# #         # Use a 4-second window for Welch
# #         if self.window_samples is None:
# #             self.window_samples = self.sampling_frequency * 4

# # class FeatureExtractor:
# #     def __init__(self, config: dict):
# #         """
# #         Initialize the feature extractor.
# #         Expects a config dictionary with keys:
# #             'preprocessing': {'sampling_frequency': int},
# #             'features': {'spectral': {'bands': dict}}
# #         """
# #         self.config = config
# #         self.sampling_frequency = config['preprocessing']['sampling_frequency']
# #         self.spectral_config = SpectralConfig(
# #             bands=config['features']['spectral']['bands'],
# #             sampling_frequency=self.sampling_frequency
# #         )
    
# #     def extract_features_from_epoch(self, epoch_df: pd.DataFrame) -> pd.DataFrame:
# #         """
# #         For the given epoch DataFrame (each row = one electrode), compute features for each electrode.
# #         The returned DataFrame will have all original metadata columns (unchanged) and new feature columns appended.
# #         """
# #         feature_rows = []
# #         # Iterate over each electrode (row)
# #         for idx, row in epoch_df.iterrows():
# #             data = row['data']  # The EEG signal (1D numpy array)
            
# #             # 1) PSD-based features using a 4-second Welch window
# #             psd_features = self._compute_normalized_psd(data)
# #             # 2) Entropy computed via 5-second nonoverlapping windows, then averaged
# #             entropy_features = self._compute_entropy_fullts(data, window_size_sec=5, stride_sec=5)
            
# #             # Combine computed features into a single dictionary
# #             new_features = {**psd_features, **entropy_features}
            
# #             # Create a combined row: copy original metadata columns (all columns from epoch_df)
# #             # then append the new features.
# #             combined = row.to_dict()
# #             # If desired, you may drop the raw 'data' column:
# #             # combined.pop('data', None)
# #             combined.update(new_features)
# #             feature_rows.append(combined)
# #         return pd.DataFrame(feature_rows)
    
# #     def _compute_normalized_psd(self, data: np.ndarray) -> dict:
# #         fs = self.sampling_frequency
# #         window_samples = self.spectral_config.window_samples  # 4-second window
# #         noverlap = window_samples // 2
        
# #         window = np.hamming(window_samples)
# #         # Use np.trapezoid (instead of np.trapz) to remove deprecation warning
# #         f, pxx = welch(data, fs=fs, window=window, nperseg=window_samples, noverlap=noverlap, scaling='density')
# #         mask = (f < 57.5) | (f > 62.5)
# #         f = f[mask]
# #         pxx = pxx[mask]
        
# #         band_powers = {}
# #         for band_name, freq_range in self.spectral_config.bands.items():
# #             idx = (f >= freq_range[0]) & (f <= freq_range[1])
# #             power = np.trapezoid(pxx[idx], f[idx])
# #             band_powers[band_name] = power
        
# #         total_power = sum(band_powers.values()) + 1e-8
        
# #         features = {}
# #         for band_name, power in band_powers.items():
# #             features[f'{band_name}_power'] = power
# #             features[f'{band_name}_rel'] = power / total_power
# #             features[f'{band_name}_log'] = np.log10(power + 1)
# #         return features
    
# #     def _compute_entropy_fullts(self, data: np.ndarray, window_size_sec: float, stride_sec: float) -> dict:
# #         fs = self.sampling_frequency
# #         window_samples = int(window_size_sec * fs)
# #         stride_samples = int(stride_sec * fs)
# #         n_samples = len(data)
        
# #         entropies = []
# #         nan_count = 0  # count problematic windows
# #         total_windows = 0
        
# #         start = 0
# #         while start + window_samples <= n_samples:
# #             total_windows += 1
# #             segment = data[start : start + window_samples]
# #             segment_filt = self._apply_filters(segment)
# #             # Replace any non-finite values with zero
# #             segment_filt = np.nan_to_num(segment_filt, nan=0.0, posinf=0.0, neginf=0.0)
            
# #             # Check if the segment is all zero
# #             if np.all(segment_filt == 0):
# #                 nan_count += 1
# #                 start += stride_samples
# #                 continue
            
# #             try:
# #                 hist, _ = np.histogram(segment_filt, bins='auto', density=True)
# #             except ValueError:
# #                 nan_count += 1
# #                 start += stride_samples
# #                 continue
            
# #             if hist.sum() == 0:
# #                 nan_count += 1
# #                 start += stride_samples
# #                 continue

# #             p = hist[hist > 0]
# #             p = p / p.sum()
# #             H = -np.sum(p * np.log2(p))
# #             entropies.append(H)
# #             start += stride_samples
        
# #         # Print a single summary message per electrode
# #         print(f"Entropy: Processed {total_windows} windows; {nan_count} windows skipped due to invalid data.")
        
# #         mean_entropy = np.mean(entropies) if entropies else 0
# #         entropy_feature = np.log10(mean_entropy + 1)
# #         return {'entropy_5secwin': entropy_feature}

    
# #     def _apply_filters(self, data: np.ndarray) -> np.ndarray:
# #         fs = self.sampling_frequency
# #         data = np.asarray(data).flatten()
        
# #         b, a = butter(3, 80/(fs/2), btype='low')
# #         data = filtfilt(b, a, data)
        
# #         b, a = butter(3, 1/(fs/2), btype='high')
# #         data = filtfilt(b, a, data)
        
# #         b, a = iirnotch(60, 30, fs)
# #         data = filtfilt(b, a, data)
        
# #         return data

# # def run_feature_extraction(base_path, config):
# #     """
# #     Runs the feature extraction pipeline on the clean HUP dataset.
# #     For each subject and each epoch:
# #       - Loads the epoch's metadata table (each row is one electrode).
# #       - Computes new features for each electrode and appends them to the original metadata.
# #       - Saves the resulting DataFrame as a new pickle file "metadata_and_features_epch{epoch_idx}.pkl"
# #         in the same subject directory.
# #       - Also saves an associated CSV file "metadata_and_features_epch{epoch_idx}.csv" with the same structure.
# #     """
# #     file_paths_dict = get_clean_hup_file_paths(base_path)
# #     extractor = FeatureExtractor(config)
    
# #     for subject, epochs_dict in file_paths_dict.items():
# #         subject_dir = os.path.join(base_path, subject)
# #         for epoch_idx, file_path in epochs_dict.items():
# #             # Load the metadata table for the epoch
# #             epoch_df = load_epoch(file_path)
# #             # Compute features per electrode (each row)
# #             features_df = extractor.extract_features_from_epoch(epoch_df)
# #             # Save as pickle file - avoid dupes
# #             out_pkl_filename = f"metadata_and_features_epch{epoch_idx}.pkl"
# #             out_pkl_path = os.path.join(subject_dir, out_pkl_filename)
# #             if not os.path.exists(out_pkl_path):
# #                 features_df.to_pickle(out_pkl_path)
# #                 logger.info(f"Saved features for {subject}, epoch {epoch_idx} to {out_pkl_path}")
# #             else:
# #                 logger.info(f"File {out_pkl_path} already exists; skipping.")
# #             # out_pkl_filename = f"metadata_and_features_epch{epoch_idx}.pkl"
# #             # out_pkl_path = os.path.join(subject_dir, out_pkl_filename)
# #             # features_df.to_pickle(out_pkl_path)
# #             # logger.info(f"Saved features for {subject}, epoch {epoch_idx} to {out_pkl_path}")
            
# #             # Save the entire DataFrame to CSV
            
# #             out_csv_filename = f"metadata_and_features_epch{epoch_idx}.csv"
# #             out_csv_path = os.path.join(subject_dir, out_csv_filename)
# #             features_df_reset = features_df.reset_index()
# #             features_df_reset.to_csv(out_csv_path, index=False)
# #             logger.info(f"Saved features (with electrode labels as a column) for {subject}, epoch {epoch_idx} to {out_csv_path}")
# #             # features_df.to_csv(out_csv_path, index=False)
# #             # logger.info(f"Saved features for {subject}, epoch {epoch_idx} to {out_csv_path}")

# # if __name__ == '__main__':
# #     # Define configuration parameters
# #     config = {
# #         'preprocessing': {
# #             'sampling_frequency': 200  # Set to your sampling frequency
# #         },
# #         'features': {
# #             'spectral': {
# #                 'bands': {
# #                     'delta': [0.5, 4],
# #                     'theta': [4, 8],
# #                     'alpha': [8, 12],
# #                     'beta': [12, 30],
# #                     'gamma': [30, 80]
# #                 }
# #             }
# #         }
# #     }
    
# #     # Base path to the clean HUP data (subject folders here)
# #     base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
    
# #     # Run the extraction process one subject at a time
# #     run_feature_extraction(base_path, config)
