import logging
import pandas as pd
import scipy.io as sio
import numpy as np
from pathlib import Path
from filter_by_outcome import load_and_process_outcomes, filter_features_by_outcome

# Define paths
RESULTS_DIR = Path('../results')
DATA_DIR = Path('../Data')

PATHS = {
    'electrode_features': RESULTS_DIR / 'hup_electrode_level_features.csv',
    'region_features': RESULTS_DIR / 'hup_region_level_features.csv',
    'region_averages': RESULTS_DIR / 'hup_region_averages_features.csv',
    'patient_map': RESULTS_DIR / 'hup_patient_map.csv',
    'patient_arr': RESULTS_DIR / 'hup_idx_map_arr.npy',
    'hup_atlas': DATA_DIR / 'HUP_atlas.mat',
    'metadata': DATA_DIR / 'metaData.csv',
    # Output paths
    'go_electrode_features': RESULTS_DIR / 'go_hup_electrode_level_features.csv',
    'go_region_features': RESULTS_DIR / 'go_hup_region_level_features.csv',
    'go_region_averages': RESULTS_DIR / 'go_hup_region_averages_features.csv'
}

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load original HUP features
        logger.info("Loading features...")
        electrode_features = pd.read_csv(PATHS['electrode_features'])
        logger.info(f"Loaded electrode features: {electrode_features.shape}")
        
        region_features = pd.read_csv(PATHS['region_features'])
        logger.info(f"Loaded region features: {region_features.shape}")
        logger.info(f"Region features columns: {region_features.columns.tolist()}")
        
        region_averages = pd.read_csv(PATHS['region_averages'])
        logger.info(f"Loaded region averages: {region_averages.shape}")

        # Load and process outcomes
        logger.info("Processing outcomes...")
        outcomes, good_outcome_patients = load_and_process_outcomes(
            PATHS['metadata'],
            PATHS['hup_atlas']
        )
        logger.info(f"Processed outcomes. Good outcome patients: {sorted(good_outcome_patients)}")

        # Create patient mapping directly from atlas
        logger.info("Creating patient mapping...")
        hup_atlas = sio.loadmat(PATHS['hup_atlas'])
        patient_nos = hup_atlas['patient_no'].flatten()
        
        # Create both dictionary and array mappings
        patient_map_dict = {idx: pat_no for idx, pat_no in enumerate(patient_nos)}
        patient_map_arr = np.array([num for num in patient_map_dict.values()])
        
        logger.info(f"Created mapping for {len(patient_map_dict)} electrodes")
        
        # Save both mapping formats
        pd.DataFrame({
            'electrode_idx': list(patient_map_dict.keys()),
            'patient_no': list(patient_map_dict.values())
        }).to_csv(PATHS['patient_map'], index=False)
        np.save(PATHS['patient_arr'], patient_map_arr)
        logger.info("Saved patient mappings")

        # Filter features
        logger.info("Filtering features...")
        
        # Filter electrode-level features
        go_electrode_features = filter_features_by_outcome(
            electrode_features, 
            patient_map_dict, 
            good_outcome_patients,
            level='electrode'
        )
        logger.info(f"Created good outcome electrode features: {go_electrode_features.shape}")
        
        # Filter region-level features
        go_region_features = filter_features_by_outcome(
            region_features, 
            patient_map_dict, 
            good_outcome_patients,
            level='region'
        )
        logger.info(f"Created good outcome region features: {go_region_features.shape}")
        
        # Filter region averages
        go_region_averages = filter_features_by_outcome(
            region_averages,
            patient_map_dict,
            good_outcome_patients,
            level='region'
        )
        logger.info(f"Created good outcome region averages: {go_region_averages.shape}")

        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save all filtered results
        logger.info("Saving results...")
        go_electrode_features.to_csv(PATHS['go_electrode_features'], index=False)
        go_region_features.to_csv(PATHS['go_region_features'], index=False)
        go_region_averages.to_csv(PATHS['go_region_averages'], index=False)
        
        logger.info("Successfully completed outcome filtering")
        
    except Exception as e:
        logger.error(f"Error during outcome filtering: {str(e)}")
        raise

if __name__ == "__main__":
    main()