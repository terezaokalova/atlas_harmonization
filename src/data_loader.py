# data_loader.py
import scipy.io as sio
import pandas as pd
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

@dataclass
class CohortData:
    """Container for cohort-specific data"""
    prefix: str
    time_series: pd.DataFrame
    patients: pd.DataFrame
    atlas: dict
    patient_map: dict
    region_df: pd.DataFrame
    sampling_frequency: int
    electrode_info: Optional[Dict[str, np.ndarray]] = None

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.cohorts = {}
        self.logger = logging.getLogger(__name__)
        
        # Load common atlas data
        self.dk_atlas_df = self._load_dk_atlas()
    
    def _load_dk_atlas(self) -> pd.DataFrame:
        """Load Desikan-Killiany atlas data"""
        dk_path = os.path.join(self.config['paths']['base_data'], 'desikanKilliany.csv')
        try:
            return pd.read_csv(dk_path)
        except FileNotFoundError:
            self.logger.error(f"DK atlas file not found at {dk_path}")
            raise
    
    def _get_good_electrodes(self, atlas: Dict) -> Tuple[Set[int], Dict[str, np.ndarray]]:
        """
        Identify good electrodes (non-resected, non-SOZ)
        
        Args:
            atlas: Dictionary containing atlas data
            
        Returns:
            Tuple of (good_electrode_indices, electrode_info_dict)
        """
        # Get and flatten electrode information
        resected = atlas['resected_ch'].flatten().astype(bool)
        soz = atlas['soz_ch'].flatten().astype(bool)
        
        # Identify good electrodes (neither resected nor SOZ)
        good_indices = np.where(~(resected | soz))[0]
        
        electrode_info = {
            'resected': resected,
            'soz': soz,
            'good_indices': good_indices,
            'is_good': ~(resected | soz)
        }
        
        self.logger.info(f"Total electrodes: {len(resected)}")
        self.logger.info(f"Resected electrodes: {np.sum(resected)}")
        self.logger.info(f"SOZ electrodes: {np.sum(soz)}")
        self.logger.info(f"Good electrodes: {len(good_indices)}")
        
        return set(good_indices), electrode_info
    
    def _detect_data_columns(self, atlas: Dict, prefix: str) -> tuple:
        """
        Detect patient and time series column names in atlas data
        Returns tuple of (patient_col, ts_col)
        """
        # Patient column detection
        patient_options = ['patient_no', 'Patient']
        patient_col = next((col for col in patient_options if col in atlas), None)
        
        # Time series column detection
        ts_options = ['wake_clip', 'Data_W']
        ts_col = next((col for col in ts_options if col in atlas), None)
        
        if patient_col is None or ts_col is None:
            self.logger.warning(f"Column auto-detection failed for {prefix}")
            print(f"\nAvailable columns in {prefix}_atlas.mat:")
            print([key for key in atlas.keys() if not key.startswith('__')])
            
            if patient_col is None:
                patient_col = input("Enter the column name for patient IDs: ").strip()
            if ts_col is None:
                ts_col = input("Enter the column name for time series data: ").strip()
                
            # Verify input columns exist
            if patient_col not in atlas or ts_col not in atlas:
                raise ValueError(f"Specified columns not found in atlas: {patient_col}, {ts_col}")
        
        return patient_col, ts_col
    
    def _get_sampling_frequency(self, atlas: Dict) -> int:
        """Extract sampling frequency from atlas data"""
        try:
            return int(atlas['SamplingFrequency'].flatten()[
                ~np.isnan(atlas['SamplingFrequency'].flatten())
            ][0])
        except (KeyError, IndexError):
            self.logger.warning("Using default sampling frequency from config")
            return self.config['preprocessing']['sampling_frequency']
    
    def _validate_data(self, time_series: pd.DataFrame, patient_data: pd.DataFrame) -> None:
        """Validate loaded data"""
        if time_series.empty:
            raise ValueError("Time series data is empty")
        if patient_data.empty:
            raise ValueError("Patient data is empty")
        if time_series.shape[1] != len(patient_data):
            raise ValueError("Mismatch between time series and patient data dimensions")
    
    def load_cohort(self, prefix: str) -> CohortData:
        """
        Load data for a specific cohort
        Args:
            prefix: Cohort identifier (e.g., 'hup', 'mni')
        Returns:
            CohortData object containing all cohort data
        """
        self.logger.info(f"Loading {prefix} cohort data")
        
        try:
            # Load atlas and region data
            atlas_path = os.path.join(self.config['paths']['base_data'], f'{prefix}_atlas.mat')
            region_path = os.path.join(self.config['paths']['base_data'], f'{prefix}_df.csv')
            
            atlas = sio.loadmat(atlas_path)
            region_df = pd.read_csv(region_path)
            
            # Detect column names
            patient_col, ts_col = self._detect_data_columns(atlas, prefix)
            
            # Extract data
            time_series = pd.DataFrame(atlas[ts_col])
            patients = pd.DataFrame(atlas[patient_col])
            
            # Create patient mapping
            patient_nums = atlas[patient_col].flatten()
            patient_map = {idx: num for idx, num in enumerate(patient_nums)}
            
            # Get sampling frequency
            sampling_freq = self._get_sampling_frequency(atlas)
            
            # Get electrode information for HUP cohort
            electrode_info = None
            if prefix == 'hup':
                _, electrode_info = self._get_good_electrodes(atlas)
            
            # Create CohortData object
            cohort = CohortData(
                prefix=prefix,
                time_series=time_series,
                patients=patients,
                atlas=atlas,
                patient_map=patient_map,
                region_df=region_df,
                sampling_frequency=sampling_freq,
                electrode_info=electrode_info
            )
            
            # Validate data
            self._validate_data(cohort.time_series, cohort.patients)
            
            self.cohorts[prefix] = cohort
            self.logger.info(f"Successfully loaded {prefix} cohort data")
            
            return cohort
            
        except Exception as e:
            self.logger.error(f"Error loading {prefix} cohort: {str(e)}")
            raise
    
    def get_patient_ids(self, cohort: CohortData) -> np.ndarray:
        """Get unique patient IDs for a cohort"""
        return np.unique(cohort.patients)
    
    def get_electrode_count(self, cohort: CohortData) -> int:
        """Get total electrode count for a cohort"""
        return len(cohort.patient_map)