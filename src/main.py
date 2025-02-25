# main.py
import yaml
import os
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
# from feature_extractor_added_fooof import FeatureExtractor
from region_aggregator import RegionAggregator
from utils import setup_logging, validate_paths

class Pipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline with configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate paths
            validate_paths(self.config)
            
            # Setup logging
            setup_logging(self.config)
            self.logger = logging.getLogger(__name__)
            
            # Initialize components
            self.data_loader = DataLoader(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            self.region_aggregator = RegionAggregator(self.config)
            
        except Exception as e:
            raise RuntimeError(f"Pipeline initialization failed: {str(e)}")
    
    def save_patient_mapping(self, cohort, prefix: str, force: bool = False):
        """Save patient to electrode mapping in both CSV and NPY formats for both full and filtered datasets"""
        # Original mapping paths
        map_csv_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_patient_map.csv'
        )
        map_arr_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_idx_map_arr.npy'
        )
        
        # Good electrode mapping paths
        ge_map_csv_path = os.path.join(
            self.config['paths']['results'],
            f'ge_{prefix}_patient_map.csv'
        )
        ge_map_arr_path = os.path.join(
            self.config['paths']['results'],
            f'ge_{prefix}_idx_map_arr.npy'
        )
        
        if not os.path.exists(map_csv_path) or force:
            # Save original mappings
            patient_map_df = pd.DataFrame({
                'electrode_idx': range(len(cohort.patient_map)),
                'patient_no': list(cohort.patient_map.values())
            })
            patient_map_df.to_csv(map_csv_path, index=False)
            
            idx_map_arr = np.array([num for num in cohort.patient_map.values()])
            np.save(map_arr_path, idx_map_arr)
            
            self.logger.info(f"Saved {prefix} mappings to {map_csv_path} and {map_arr_path}")
            
            # For HUP cohort, also save good electrode mappings
            if prefix == 'hup' and cohort.electrode_info is not None:
                good_indices = cohort.electrode_info['good_indices']
                ge_patient_map_df = pd.DataFrame({
                    'electrode_idx': range(len(good_indices)),
                    'patient_no': [cohort.patient_map[idx] for idx in good_indices]
                })
                ge_patient_map_df.to_csv(ge_map_csv_path, index=False)
                
                ge_idx_map_arr = np.array([cohort.patient_map[idx] for idx in good_indices])
                np.save(ge_map_arr_path, ge_idx_map_arr)
                
                self.logger.info(f"Saved good electrode mappings to {ge_map_csv_path} and {ge_map_arr_path}")
    
    def run(self, cohorts: list = None, force_compute: bool = False):
        """
        Run the complete pipeline
        
        Args:
            cohorts: List of cohort prefixes to process
            force_compute: If True, recompute all features even if they exist
        """
        if cohorts is None:
            cohorts = ['hup', 'mni']
        
        results = {}
        
        for prefix in cohorts:
            try:
                self.logger.info(f"Processing {prefix} cohort")
                
                # Load data
                cohort = self.data_loader.load_cohort(prefix)
                
                # Save patient mapping
                self.save_patient_mapping(cohort, prefix, force=force_compute)
                
                # Extract features (returns dict with 'full' and possibly 'filtered')
                self.logger.info(f"Extracting features for {prefix}")
                features = self.feature_extractor.extract_cohort_features(cohort)
                
                # Process regions and save results
                self.logger.info(f"Processing regions for {prefix}")
                results[prefix] = self.region_aggregator.process_features(
                    features, 
                    cohort,
                    self.data_loader.dk_atlas_df
                )
                
                # Save results
                self.region_aggregator.save_results(prefix, results[prefix])
                
                self.logger.info(f"Completed processing {prefix} cohort")
                
            except Exception as e:
                self.logger.error(f"Error processing {prefix} cohort: {str(e)}")
                raise
        
        return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--force', action='store_true', 
                       help='Force recomputation of all features')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--cohorts', nargs='+', default=['hup', 'mni'],
                       help='Cohorts to process')
    return parser.parse_args()

def main():
    # Initialize logger first
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize and run pipeline
    try:
        pipeline = Pipeline(config_path=args.config)
        results = pipeline.run(
            cohorts=args.cohorts,
            force_compute=args.force
        )
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()