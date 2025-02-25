# main_outcome_filter.py
import logging
import pandas as pd
import scipy.io as sio
from pathlib import Path
from filter_by_outcome import load_and_process_outcomes, filter_features_by_outcome

# Define paths
RESULTS_DIR = Path('../results')
DATA_DIR = Path('../Data')

PATHS = {
    # Original HUP files
    'hup_electrode': RESULTS_DIR / 'hup_electrode_features.csv',
    'hup_region': RESULTS_DIR / 'hup_region_features.csv',
    'hup_averages': RESULTS_DIR / 'hup_region_averages.csv',
    
    # Good electrode (ge) HUP files
    'ge_hup_electrode': RESULTS_DIR / 'ge_hup_electrode_features.csv',
    'ge_hup_region': RESULTS_DIR / 'ge_hup_region_features.csv',
    'ge_hup_averages': RESULTS_DIR / 'ge_hup_region_averages.csv',
    
    # Patient mappings
    'hup_patient_map': RESULTS_DIR / 'hup_patient_map.csv',
    'ge_hup_patient_map': RESULTS_DIR / 'ge_hup_patient_map.csv',
    
    # Atlas and metadata
    'hup_atlas': DATA_DIR / 'HUP_atlas.mat',
    'metadata': DATA_DIR / 'metaData.csv',
    
    # Output paths for original data
    'go_hup_electrode': RESULTS_DIR / 'go_hup_electrode_features.csv',
    'go_hup_region': RESULTS_DIR / 'go_hup_region_features.csv',
    'go_hup_averages': RESULTS_DIR / 'go_hup_region_averages.csv',
    
    # Output paths for good electrode data
    'ge_go_hup_electrode': RESULTS_DIR / 'ge_go_hup_electrode_features.csv',
    'ge_go_hup_region': RESULTS_DIR / 'ge_go_hup_region_features.csv',
    'ge_go_hup_averages': RESULTS_DIR / 'ge_go_hup_region_averages.csv'
}

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def process_dataset(features_df: pd.DataFrame, 
                   patient_map_dict: dict,
                   good_outcome_patients: set,
                   level: str,
                   dataset_type: str,
                   is_ge: bool = False) -> pd.DataFrame:
    """Process a single dataset with logging"""
    logger = logging.getLogger(__name__)
    
    filtered_df = filter_features_by_outcome(
        features_df, 
        patient_map_dict,
        good_outcome_patients,
        level=level,
        is_ge=is_ge
    )
    
    logger.info(f"Created {dataset_type} features: {filtered_df.shape}")
    return filtered_df

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load and process outcomes
        logger.info("Processing outcomes...")
        outcomes, good_outcome_patients = load_and_process_outcomes(
            PATHS['metadata'],
            PATHS['hup_atlas']
        )
        
        # Create patient mapping from atlas
        logger.info("Creating patient mapping...")
        hup_atlas = sio.loadmat(PATHS['hup_atlas'])
        patient_nos = hup_atlas['patient_no'].flatten()
        patient_map_dict = {idx: pat_no for idx, pat_no in enumerate(patient_nos)}
        
        # Process original HUP files
        logger.info("Processing original HUP files...")
        datasets = {
            ('electrode', 'electrode_level'): (PATHS['hup_electrode'], PATHS['go_hup_electrode']),
            ('region', 'region_level'): (PATHS['hup_region'], PATHS['go_hup_region']),
            ('region_averages', 'region_averages'): (PATHS['hup_averages'], PATHS['go_hup_averages'])
        }
        
        for (level, stage), (input_path, output_path) in datasets.items():
            logger.info(f"Processing {stage}...")
            try:
                features_df = pd.read_csv(input_path)
                filtered_df = process_dataset(
                    features_df,
                    patient_map_dict,
                    good_outcome_patients,
                    level=level,
                    dataset_type=f"good outcome {stage}",
                    is_ge=False
                )
                filtered_df.to_csv(output_path, index=False)
            except Exception as e:
                logger.error(f"Error processing {stage}: {str(e)}")
                continue
        
        # Process good electrode (ge) HUP files
        logger.info("Processing good electrode HUP files...")
        ge_datasets = {
            ('electrode', 'electrode_level'): (PATHS['ge_hup_electrode'], PATHS['ge_go_hup_electrode']),
            ('region', 'region_level'): (PATHS['ge_hup_region'], PATHS['ge_go_hup_region']),
            ('region_averages', 'region_averages'): (PATHS['ge_hup_averages'], PATHS['ge_go_hup_averages'])
        }
        
        for (level, stage), (input_path, output_path) in ge_datasets.items():
            logger.info(f"Processing good electrode {stage}...")
            try:
                features_df = pd.read_csv(input_path)
                filtered_df = process_dataset(
                    features_df,
                    patient_map_dict,
                    good_outcome_patients,
                    level=level,
                    dataset_type=f"good electrode good outcome {stage}",
                    is_ge=True
                )
                filtered_df.to_csv(output_path, index=False)
            except Exception as e:
                logger.error(f"Error processing good electrode {stage}: {str(e)}")
                continue
        
        logger.info("Successfully completed outcome filtering")
        
    except Exception as e:
        logger.error(f"Error during filtering: {str(e)}")
        raise

if __name__ == "__main__":
    main()