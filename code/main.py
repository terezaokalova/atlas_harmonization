import yaml
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from region_aggregator import RegionAggregator
from utils import setup_logging

class Pipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline with configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure config.yaml exists in the same directory as main.py")
        
        # Setup logging
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.region_aggregator = RegionAggregator(self.config)
        
        # Create results directory
        Path(self.config['paths']['results']).mkdir(parents=True, exist_ok=True)
    
    def _file_exists(self, prefix: str, stage: str) -> bool:
        """Check if output file already exists"""
        output_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_{stage}_features.csv'
        )
        return os.path.exists(output_path)
    
    def save_results(self, data: pd.DataFrame, prefix: str, stage: str, force: bool = False):
        """Save results with proper naming convention"""
        output_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_{stage}_features.csv'
        )
        
        if not os.path.exists(output_path) or force:
            data.to_csv(output_path, index=False)
            self.logger.info(f"Saved {stage} features for {prefix} to {output_path}")
        else:
            self.logger.info(f"Skipping save: {stage} features for {prefix} already exist")
    
    def save_patient_mapping(self, cohort, prefix: str, force: bool = False):
        """Save patient to electrode mapping in both CSV and NPY formats"""
        # Paths for both formats
        map_csv_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_patient_map.csv'
        )
        map_arr_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_idx_map_arr.npy'  # Consistent with original naming
        )
        
        if not os.path.exists(map_csv_path) or force:
            # Create and save CSV format
            patient_map_df = pd.DataFrame({
                'electrode_idx': range(len(cohort.patient_map)),
                'patient_no': list(cohort.patient_map.values())
            })
            patient_map_df.to_csv(map_csv_path, index=False)
            
            # Create and save NPY format
            idx_map_arr = np.array([num for num in cohort.patient_map.values()])
            np.save(map_arr_path, idx_map_arr)
            
            self.logger.info(f"Saved {prefix} mappings to {map_csv_path} and {map_arr_path}")

    def run(self, cohorts: list = None, force_compute: bool = False):
        """
        Run the complete pipeline
        
        Args:
            cohorts: List of cohorts to process
            force_compute: If True, recompute everything even if files exist
        """
        if cohorts is None:
            cohorts = ['hup', 'mni']
        
        results = {}
        
        for prefix in cohorts:
            try:
                self.logger.info(f"Processing {prefix} cohort")
                
                # Check if all files exist
                stages = ['electrode_level', 'region_level', 'region_averages']
                all_exist = all(self._file_exists(prefix, stage) for stage in stages)
                
                if all_exist and not force_compute:
                    self.logger.info(f"Skipping {prefix} cohort - all files exist")
                    continue
                
                # Load data
                cohort = self.data_loader.load_cohort(prefix)
                
                # Save patient mapping for HUP
                self.save_patient_mapping(cohort, prefix, force=force_compute)
                
                # Extract features
                self.logger.info(f"Extracting features for {prefix}")
                features = self.feature_extractor.extract_cohort_features(cohort)
                
                # Save electrode-level features
                self.save_results(features, prefix, 'electrode_level', force=force_compute)
                
                # Aggregate by region
                self.logger.info(f"Aggregating features by region for {prefix}")
                region_features = self.region_aggregator.aggregate_features_by_region(
                    features,
                    cohort.region_df,
                    self.data_loader.dk_atlas_df,
                    list(cohort.patient_map.values())
                )
                self.save_results(region_features, prefix, 'region_level', force=force_compute)
                
                # Average across regions
                self.logger.info(f"Computing region averages for {prefix}")
                region_averages = self.region_aggregator.average_by_region(
                    features,
                    cohort.region_df,
                    self.data_loader.dk_atlas_df,
                    list(cohort.patient_map.values())
                )
                self.save_results(region_averages, prefix, 'region_averages', force=force_compute)
                
                results[prefix] = {
                    'electrode_features': features,
                    'region_features': region_features,
                    'region_averages': region_averages
                }
                
                self.logger.info(f"Completed processing {prefix} cohort")
                
            except Exception as e:
                self.logger.error(f"Error processing {prefix} cohort: {str(e)}")
                raise
        
        return results

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--force', action='store_true', help='Force recomputation of all features')
    args = parser.parse_args()

    # Run pipeline with force option from command line
    pipeline = Pipeline()
    results = pipeline.run(force_compute=args.force)
    
    logger = logging.getLogger(__name__)
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()