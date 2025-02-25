import numpy as np 
import pandas as pd
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.signal.windows import hamming
from scipy.stats import entropy
import logging
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
from fooof import FOOOF

@dataclass
class SpectralConfig:
    """Configuration for spectral analysis"""
    bands: Dict[str, List[float]]
    sampling_frequency: int
    window_samples: int = None
    
    def __post_init__(self):
        if self.window_samples is None:
            self.window_samples = self.sampling_frequency * 2  # 2-second window

@dataclass 
class FOOOFParams:
    """Parameters and results from FOOOF analysis"""
    offset: float
    exponent: float
    alpha_cf: float  # Center frequency
    alpha_amp: float # Amplitude
    alpha_bw: float  # Bandwidth
    
    @classmethod
    def create_empty(cls) -> 'FOOOFParams':
        """Create a FOOOFParams instance with NaN values"""
        return cls(
            offset=np.nan,
            exponent=np.nan, 
            alpha_cf=np.nan,
            alpha_amp=np.nan,
            alpha_bw=np.nan
        )

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

    def _compute_fooof_features(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        freq_range: Optional[List[float]] = None
    ) -> FOOOFParams:
        """
        Compute FOOOF features from frequency and power data.
        
        Args:
            freqs: Frequency values
            psd: Power spectral density values
            freq_range: Optional frequency range for fitting [min, max]
            
        Returns:
            FOOOFParams object containing extracted parameters
        """
        try:
            # Ensure inputs are properly formatted
            freqs = np.asarray(freqs, dtype=float)
            psd = np.asarray(psd, dtype=float)
            
            # Remove any zeros or negative values 
            mask = psd > 0
            freqs = freqs[mask]
            psd = psd[mask]
            
            # Set default frequency range if not provided
            if freq_range is None:
                freq_range = [1, 80]
            
            # Initialize and fit FOOOF model
            fm = FOOOF(
                peak_width_limits=[1.0, 8.0],
                max_n_peaks=6,
                min_peak_height=0.1,
                peak_threshold=2.0,
                aperiodic_mode='fixed'
            )
            
            fm.fit(freqs, psd, freq_range)
            
            # Extract parameters
            aperiodic_params = fm.get_params('aperiodic_params')
            peak_params = fm.get_params('peak_params')
            
            # Find alpha peak (8-13 Hz)
            alpha_peak = [pk for pk in peak_params if 8 <= pk[0] <= 13]
            alpha_params = alpha_peak[0] if alpha_peak else (np.nan, np.nan, np.nan)
            
            return FOOOFParams(
                offset=aperiodic_params[0],
                exponent=aperiodic_params[1],
                alpha_cf=alpha_params[0],
                alpha_amp=alpha_params[1],
                alpha_bw=alpha_params[2]
            )
            
        except Exception as e:
            self.logger.error(f"FOOOF analysis failed: {str(e)}")
            return FOOOFParams.create_empty()

    def extract_cohort_features(self, cohort_data) -> Dict[str, pd.DataFrame]:
        """Extract features for both full and filtered datasets"""
        self.logger.info(f"Extracting features for {cohort_data.prefix} cohort")
        features = {}
        
        # Process full dataset
        self.logger.info("Processing full dataset...")
        features['full'] = self._extract_features(
            cohort_data.time_series,
            cohort_data.patients,
            cohort_data.patient_map,
            cohort_data.sampling_frequency
        )
        
        # Process filtered dataset if electrode info exists
        if cohort_data.electrode_info is not None:
            self.logger.info("Processing filtered dataset...")
            good_indices = cohort_data.electrode_info['good_indices']
            filtered_ts = cohort_data.time_series.iloc[:, list(good_indices)]
            
            filtered_map = {new_idx: cohort_data.patient_map[old_idx] 
                          for new_idx, old_idx in enumerate(good_indices)}
            
            features['filtered'] = self._extract_features(
                filtered_ts,
                pd.DataFrame([cohort_data.patient_map[idx] for idx in good_indices]),
                filtered_map,
                cohort_data.sampling_frequency
            )
        
        return features

    def _extract_features(self, time_series: pd.DataFrame,
                         patients: pd.DataFrame,
                         patient_map: Dict,
                         sampling_frequency: int) -> pd.DataFrame:
        """Extract all features for the given data"""
        psd_features = pd.DataFrame()
        entropy_1min = pd.DataFrame()
        entropy_fullts = pd.DataFrame()
        fooof_features = pd.DataFrame()
        
        patient_ids = np.unique(patients)
        
        for patient_id in patient_ids:
            self.logger.info(f"Processing patient {patient_id}")
            patient_electrodes = np.where(
                np.array(list(patient_map.values())) == patient_id
            )[0]
            
            for idx in patient_electrodes:
                try:
                    electrode_data = time_series.iloc[:, idx].values
                    
                    # Original PSD features
                    psd_features = self._compute_normalized_psd(
                        psd_features,
                        electrode_data,
                        sampling_frequency
                    )
                    
                    # Compute Welch PSD for FOOOF
                    f, psd = welch(electrode_data, fs=sampling_frequency,
                                 window=hamming(self.spectral_config.window_samples),
                                 nfft=self.spectral_config.window_samples,
                                 scaling='density',
                                 noverlap=self.spectral_config.window_samples//2)
                    
                    # FOOOF analysis
                    fooof_params = self._compute_fooof_features(f, psd)
                    fooof_features = pd.concat([
                        fooof_features,
                        pd.DataFrame([{
                            'fooof_offset': fooof_params.offset,
                            'fooof_exponent': fooof_params.exponent,
                            'fooof_alpha_cf': fooof_params.alpha_cf,
                            'fooof_alpha_amp': fooof_params.alpha_amp,
                            'fooof_alpha_bw': fooof_params.alpha_bw
                        }])
                    ], ignore_index=True)
                    
                    # Entropy features
                    entropy_1min = self._compute_entropy_1min(
                        entropy_1min,
                        electrode_data,
                        sampling_frequency
                    )
                    
                    entropy_fullts = self._compute_entropy_fullts(
                        entropy_fullts,
                        electrode_data,
                        sampling_frequency,
                        self.config['preprocessing']['windows']['size_minutes'],
                        self.config['preprocessing']['windows']['stride_minutes']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing electrode {idx} for patient {patient_id}: {str(e)}")
                    continue
        
        # Combine all features
        combined_features = pd.concat([
            psd_features,
            fooof_features,
            entropy_1min[['entropy']].rename(columns={'entropy': 'entropy_1min'}),
            entropy_fullts[['entropy_fullts']]
        ], axis=1)
        
        return combined_features

    # The following methods remain exactly as they were in the original:
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