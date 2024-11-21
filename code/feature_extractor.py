# src/feature_extractor.py
import numpy as np
import pandas as pd
# from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.signal.windows import hamming
from scipy.stats import entropy
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class SpectralConfig:
    """Configuration for spectral analysis"""
    bands: Dict[str, List[float]]
    sampling_frequency: int
    window_samples: int = None
    
    def __post_init__(self):
        if self.window_samples is None:
            self.window_samples = self.sampling_frequency * 2  # 2-second window

class FeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.features = {}
        
        # Initialize spectral config
        self.spectral_config = SpectralConfig(
            bands=config['features']['spectral']['bands'],
            sampling_frequency=config['preprocessing']['sampling_frequency']
        )
        
    def extract_cohort_features(self, cohort_data) -> pd.DataFrame:
        """Extract all features for a cohort"""
        self.logger.info(f"Extracting features for {cohort_data.prefix} cohort")
        
        psd_features = pd.DataFrame()
        entropy_1min = pd.DataFrame()
        entropy_fullts = pd.DataFrame()
        
        patient_ids = np.unique(cohort_data.patients)
        
        for patient_id in patient_ids:
            self.logger.info(f"Processing patient {patient_id}")
            patient_electrodes = np.where(
                np.array(list(cohort_data.patient_map.values())) == patient_id
            )[0]
            
            for idx in patient_electrodes:
                electrode_data = cohort_data.time_series.iloc[:, idx].values
                
                try:
                    # Extract PSD features
                    psd_features = self._compute_normalized_psd(
                        psd_features, 
                        electrode_data, 
                        cohort_data.sampling_frequency
                    )
                    
                    # Extract entropy features
                    entropy_1min = self._compute_entropy_1min(
                        entropy_1min,
                        electrode_data,
                        cohort_data.sampling_frequency
                    )
                    
                    entropy_fullts = self._compute_entropy_fullts(
                        entropy_fullts,
                        electrode_data,
                        cohort_data.sampling_frequency,
                        self.config['preprocessing']['windows']['size_minutes'],
                        self.config['preprocessing']['windows']['stride_minutes']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing electrode {idx} for patient {patient_id}: {str(e)}")
                    continue
        
        # Combine all features
        combined_features = pd.concat([
            psd_features,
            entropy_1min[['entropy']].rename(columns={'entropy': 'entropy_1min'}),
            entropy_fullts[['entropy_fullts']]
        ], axis=1)
        
        self.logger.info(f"Completed feature extraction for {cohort_data.prefix} cohort")
        return combined_features
    
    def _compute_normalized_psd(self, features_df: pd.DataFrame, 
                              data: np.ndarray, 
                              sampling_frequency: int) -> pd.DataFrame:
        """Compute normalized power spectral density features"""
        Fs = sampling_frequency
        window = self.spectral_config.window_samples
        NFFT = window
        
        # Compute PSD
        f, data_psd = welch(data, fs=Fs, window=hamming(window),
                           nfft=NFFT, scaling='density', noverlap=window//2)
        
        # Filter out noise frequency 57.5Hz to 62.5Hz
        noise_mask = (f >= 57.5) & (f <= 62.5)
        f = f[~noise_mask]
        data_psd = data_psd[~noise_mask]
        
        def bandpower(psd: np.ndarray, freqs: np.ndarray, freq_range: Tuple[float, float]) -> float:
            """Calculate power in the given frequency range"""
            idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            return np.trapezoid(psd[idx], freqs[idx])
        
        # Calculate band powers
        band_powers = {band: bandpower(data_psd, f, freq_range)
                      for band, freq_range in self.spectral_config.bands.items()}
        
        # Compute log transform
        log_band_powers = {f'{band}log': np.log10(power + 1)
                          for band, power in band_powers.items()}
        
        # Calculate total power
        total_band_power = np.sum([value for value in log_band_powers.values()])
        
        # Calculate relative powers
        relative_band_powers = {
            f'{band}Rel': log_band_powers[f'{band}log'] / total_band_power
            for band in self.spectral_config.bands.keys()
        }
        
        # Create DataFrame row
        data_to_append = pd.DataFrame([relative_band_powers])
        return pd.concat([features_df, data_to_append], ignore_index=True)
    
    def _compute_entropy_1min(self, features_df: pd.DataFrame,
                            data: np.ndarray,
                            sampling_frequency: int) -> pd.DataFrame:
        """Compute entropy for first minute segment"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Get first minute of data
        data_seg = data[:sampling_frequency*60, :]
        
        # Apply filters
        data_seg = self._apply_filters(data_seg, sampling_frequency)
        
        # Compute Shannon entropy
        signal = data_seg[:, 0]
        hist, _ = np.histogram(signal, bins='auto', density=True)
        hist = hist / hist.sum()
        entropy_val = entropy(hist)
        
        # Log transform
        entropy_val = np.log10(entropy_val + 1)
        
        # Create new row
        data_to_append = pd.DataFrame({'entropy': [entropy_val]})
        return pd.concat([features_df, data_to_append], ignore_index=True)
    
    def _compute_entropy_fullts(self, features_df: pd.DataFrame,
                              data: np.ndarray,
                              sampling_frequency: int,
                              window_size_mins: float,
                              stride_mins: float) -> pd.DataFrame:
        """Compute entropy across full time series using sliding windows"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        # Calculate window parameters
        samples_per_window = int(sampling_frequency * 60 * window_size_mins)
        stride_samples = int(sampling_frequency * 60 * stride_mins)
        n_windows = int((len(data) - samples_per_window) // stride_samples + 1)
        
        entropies = []
        for i in range(n_windows):
            start_idx = int(i * stride_samples)
            end_idx = int(start_idx + samples_per_window)
            window_data = data[start_idx:end_idx, :]
            
            # Apply filters
            filtered_data = self._apply_filters(window_data, sampling_frequency)
            
            # Compute entropy for this window
            signal = filtered_data[:, 0]
            hist, _ = np.histogram(signal, bins='auto', density=True)
            probabilities = hist / hist.sum()
            probabilities = probabilities[probabilities > 0]
            
            # Compute Shannon entropy
            H = -np.sum(probabilities * np.log2(probabilities))
            entropies.append(H)
        
        # Store log-transformed mean entropy
        data_to_append = pd.DataFrame({
            'entropy_fullts': [np.log10(np.mean(entropies) + 1)]
        })
        
        return pd.concat([features_df, data_to_append], ignore_index=True)
    
    def _apply_filters(self, data: np.ndarray, 
                      sampling_frequency: int) -> np.ndarray:
        """Apply standard filtering pipeline to data"""
        # Low pass filter at 80Hz
        b, a = butter(3, 80/(sampling_frequency/2), btype='low')
        filtered = filtfilt(b, a, data.astype(float), axis=0)
        
        # High pass filter at 1Hz
        b, a = butter(3, 1/(sampling_frequency/2), btype='high')
        filtered = filtfilt(b, a, filtered, axis=0)
        
        # Notch filter at 60Hz
        b, a = iirnotch(60, 30, sampling_frequency)
        filtered = filtfilt(b, a, filtered, axis=0)
        
        return filtered