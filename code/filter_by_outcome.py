import pandas as pd
import numpy as np
import scipy.io as sio
import logging
from typing import Tuple, Dict, Set

logger = logging.getLogger(__name__)

def load_and_process_outcomes(meta_data_path: str, hup_atlas_path: str) -> Tuple[Dict[int, bool], Set[int]]:
    """Load metadata and create patient outcome mapping"""
    # This function remains the same
    meta_df = pd.read_csv(meta_data_path)
    hup_atlas = sio.loadmat(hup_atlas_path)
    
    patient_nos = np.unique(hup_atlas['patient_no'].flatten())
    
    outcomes = {}
    good_outcomes = set()
    
    logger.info(f"Total patients in HUP atlas: {len(patient_nos)}")
    logger.info(f"Total rows in metadata: {len(meta_df)}")
    
    for idx, row in meta_df.iterrows():
        max_engel = max(row['Engel_12_mo'], row['Engel_24_mo'])
        patient_no = patient_nos[idx]
        outcomes[patient_no] = max_engel < 1.3
        
        if max_engel < 1.3:
            good_outcomes.add(patient_no)
    
    logger.info(f"Found {len(good_outcomes)} good outcome patients")
    return outcomes, good_outcomes

def filter_features_by_outcome(features_df: pd.DataFrame,
                             patient_map: Dict[int, int],
                             good_outcome_patients: Set[int],
                             level: str = 'electrode') -> pd.DataFrame:
    """
    Filter features dataframe to keep only good outcome patients
    
    Args:
        features_df: DataFrame with features
        patient_map: Dictionary mapping indices to patient numbers
        good_outcome_patients: Set of patient numbers with good outcomes
        level: 'electrode', 'region', or 'region_averages'
    """
    logger = logging.getLogger(__name__)
    
    if level == 'electrode':
        # Original electrode-level filtering
        patient_series = pd.Series(patient_map)
        good_indices = patient_series[patient_series.isin(good_outcome_patients)].index
        filtered_df = features_df.iloc[good_indices]
        
    elif level == 'region':
        if 'patient_id' in features_df.columns:
            filtered_df = features_df[features_df['patient_id'].isin(good_outcome_patients)]
        else:
            # If no patient_id column, try roiNum and assume it's region features
            filtered_df = features_df
            logger.warning("No patient_id column found, assuming region-level features")
            
    elif level == 'region_averages':
        # Region averages are already averaged across patients, no filtering needed
        filtered_df = features_df
        logger.info("Region averages already averaged across patients, no filtering applied")
    
    else:
        raise ValueError(f"Unknown level: {level}")
    
    logger.info(f"Filtered from {len(features_df)} to {len(filtered_df)} rows")
    return filtered_df