# main.py
import yaml
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from region_aggregator import RegionAggregator
from utils import setup_logging
from data_validator import DataValidator

class Pipeline:
    def __init__(self, config_path: str = 'config.yaml'):  # Changed from 'config/config.yaml'
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
        self.validator = DataValidator()
        
        # Create results directory
        Path(self.config['paths']['results']).mkdir(parents=True, exist_ok=True)
    
    def save_results(self, data: pd.DataFrame, prefix: str, stage: str):
        """Save results with proper naming convention"""
        output_path = os.path.join(
            self.config['paths']['results'],
            f'{prefix}_{stage}_features.csv'
        )
        data.to_csv(output_path, index=False)
        self.logger.info(f"Saved {stage} features for {prefix} to {output_path}")
    
    def run(self, cohorts: list = None):
        """Run the complete pipeline"""
        if cohorts is None:
            cohorts = ['hup', 'mni']
        
        results = {}
        
        for prefix in cohorts:
            try:
                self.logger.info(f"Processing {prefix} cohort")
                
                # Load data
                cohort = self.data_loader.load_cohort(prefix)
                
                # Extract features
                self.logger.info(f"Extracting features for {prefix}")
                features = self.feature_extractor.extract_cohort_features(cohort)
                
                # Save electrode-level features
                self.save_results(features, prefix, 'electrode_level')
                
                # Aggregate by region
                self.logger.info(f"Aggregating features by region for {prefix}")
                region_features = self.region_aggregator.aggregate_features_by_region(
                    features,
                    cohort.region_df,
                    self.data_loader.dk_atlas_df,
                    list(cohort.patient_map.values())
                )
                self.save_results(region_features, prefix, 'region_level')
                
                # Average across regions
                self.logger.info(f"Computing region averages for {prefix}")
                region_averages = self.region_aggregator.average_by_region(
                    features,
                    cohort.region_df,
                    self.data_loader.dk_atlas_df,
                    list(cohort.patient_map.values())
                )
                self.save_results(region_averages, prefix, 'region_averages')
                
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

class PipelineValidator:
    def __init__(self, original_results_path: str, new_results_path: str):
        """Initialize validator with paths to original and new results"""
        self.original_path = original_results_path
        self.new_path = new_results_path
        self.logger = logging.getLogger(__name__)
    
    def load_results(self, prefix: str, stage: str) -> tuple:
        """Load both original and new results for comparison"""
        original = pd.read_csv(os.path.join(self.original_path, f'{prefix}_{stage}_features.csv'))
        new = pd.read_csv(os.path.join(self.new_path, f'{prefix}_{stage}_features.csv'))
        return original, new
    
    def compare_results(self, original: pd.DataFrame, new: pd.DataFrame,
                       rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Compare results within numerical tolerance"""
        try:
            pd.testing.assert_frame_equal(
                original, new,
                check_exact=False,
                rtol=rtol,
                atol=atol
            )
            return True
        except AssertionError as e:
            self.logger.error(f"Results comparison failed: {str(e)}")
            return False
    
    def validate_all(self, cohorts: list = None, stages: list = None) -> dict:
        """Validate all results"""
        if cohorts is None:
            cohorts = ['hup', 'mni']
        if stages is None:
            stages = ['electrode_level', 'region_level', 'region_averages']
        
        validation_results = {}
        
        for prefix in cohorts:
            validation_results[prefix] = {}
            for stage in stages:
                try:
                    original, new = self.load_results(prefix, stage)
                    matches = self.compare_results(original, new)
                    validation_results[prefix][stage] = matches
                except Exception as e:
                    self.logger.error(f"Validation failed for {prefix} {stage}: {str(e)}")
                    validation_results[prefix][stage] = False
        
        return validation_results

def main():
    # Run pipeline
    pipeline = Pipeline()
    results = pipeline.run()
    
    # Validate results against original implementation
    validator = PipelineValidator(
        original_results_path='../original_results',
        new_results_path=pipeline.config['paths']['results']
    )
    
    validation_results = validator.validate_all()
    
    # Log validation results
    logger = logging.getLogger(__name__)
    for prefix in validation_results:
        for stage, matches in validation_results[prefix].items():
            status = "matches" if matches else "differs from"
            logger.info(f"{prefix} {stage} {status} original implementation")

if __name__ == "__main__":
    main()