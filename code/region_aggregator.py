import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import os

@dataclass
class AggregationConfig:
    """Configuration for feature aggregation"""
    spectral_features: List[str] = None
    entropy_features: List[str] = None
    
    def __post_init__(self):
        if self.spectral_features is None:
            self.spectral_features = ['deltaRel', 'thetaRel', 'alphaRel', 'betaRel', 'gammaRel']
        if self.entropy_features is None:
            self.entropy_features = ['entropy_1min', 'entropy_fullts']
    
    @property
    def all_features(self) -> List[str]:
        return self.spectral_features + self.entropy_features

class RegionAggregator:
    def __init__(self, config: Dict):
        self.config = config
        self.agg_config = AggregationConfig()
        self.logger = logging.getLogger(__name__)
    
    def process_features(self, features: Dict[str, pd.DataFrame], 
                        cohort_data, 
                        dk_atlas_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process both full and filtered features
        
        Args:
            features: Dictionary containing 'full' and optionally 'filtered' DataFrames
            cohort_data: CohortData object containing cohort information
            dk_atlas_df: Desikan-Killiany atlas DataFrame
            
        Returns:
            Dictionary containing processed features for each dataset
        """
        results = {}
        
        # Process full dataset
        self.logger.info("Processing full dataset...")
        results['full'] = {
            'electrode_features': features['full'],
            'region_features': self.aggregate_features_by_region(
                features['full'],
                cohort_data.region_df,
                dk_atlas_df,
                list(cohort_data.patient_map.values())
            ),
            'region_averages': self.average_by_region(
                features['full'],
                cohort_data.region_df,
                dk_atlas_df,
                list(cohort_data.patient_map.values())
            )
        }
        
        # Process filtered dataset if it exists
        if 'filtered' in features:
            self.logger.info("Processing filtered dataset...")
            # Get good electrode indices
            good_indices = cohort_data.electrode_info['good_indices']
            filtered_region_df = cohort_data.region_df.iloc[list(good_indices)]
            
            # Create filtered patient mapping
            filtered_patient_map = {
                new_idx: cohort_data.patient_map[old_idx]
                for new_idx, old_idx in enumerate(good_indices)
            }
            
            results['filtered'] = {
                'electrode_features': features['filtered'],
                'region_features': self.aggregate_features_by_region(
                    features['filtered'],
                    filtered_region_df,
                    dk_atlas_df,
                    list(filtered_patient_map.values())
                ),
                'region_averages': self.average_by_region(
                    features['filtered'],
                    filtered_region_df,
                    dk_atlas_df,
                    list(filtered_patient_map.values())
                )
            }
        
        return results
        
    def _prepare_combined_df(self, features_df: pd.DataFrame, 
                           region_df: pd.DataFrame, 
                           patient_map_arr: np.ndarray) -> pd.DataFrame:
        """Combine features with region and patient information"""
        return pd.concat([
            features_df.reset_index(drop=True),
            pd.DataFrame({
                'roiNum': region_df['roiNum'].reset_index(drop=True),
                'patient_id': patient_map_arr
            })
        ], axis=1)
    
    def aggregate_features_by_region(self, 
                                   features_df: pd.DataFrame, 
                                   region_df: pd.DataFrame, 
                                   dk_atlas_df: pd.DataFrame, 
                                   patient_map_arr: np.ndarray) -> pd.DataFrame:
        """
        Aggregate electrode-level features to region-level features while preserving patient identity
        
        Args:
            features_df: DataFrame with electrode features
            region_df: DataFrame with electrode to region mapping
            dk_atlas_df: DataFrame with region information
            patient_map_arr: Array mapping electrodes to patient IDs
            
        Returns:
            DataFrame with aggregated features by region and patient
        """
        self.logger.info("Aggregating features by region...")
        combined_df = self._prepare_combined_df(features_df, region_df, patient_map_arr)
        
        # Group by both patient and region
        region_features = []
        
        for (pat_id, roi), group in combined_df.groupby(['patient_id', 'roiNum']):
            row_dict = {
                'patient_id': pat_id,
                'roiNum': roi
            }
            
            # Calculate means for each feature
            for feat in self.agg_config.all_features:
                if feat in group.columns:
                    row_dict[f"{feat}_mean"] = group[feat].mean()
            
            region_features.append(row_dict)
        
        # Convert to DataFrame and add region names
        region_features_df = pd.DataFrame(region_features)
        
        self.logger.info(f"Created region features with shape: {region_features_df.shape}")
        
        return pd.merge(
            region_features_df,
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )
    
    def average_by_region(self, 
                         features_df: pd.DataFrame, 
                         region_df: pd.DataFrame, 
                         dk_atlas_df: pd.DataFrame, 
                         patient_map_arr: np.ndarray) -> pd.DataFrame:
        """
        Average features across all electrodes in each region
        
        Args:
            features_df: DataFrame with electrode features
            region_df: DataFrame with electrode to region mapping
            dk_atlas_df: DataFrame with region information
            patient_map_arr: Array mapping electrodes to patient IDs
            
        Returns:
            DataFrame with averaged features by region
        """
        self.logger.info("Computing region averages...")
        combined_df = self._prepare_combined_df(features_df, region_df, patient_map_arr)
        
        # First average within patient-region
        patient_region_avg = combined_df.groupby(['patient_id', 'roiNum']).mean()
        
        # Then average across patients for each region
        region_avg = patient_region_avg.groupby('roiNum').mean()
        
        # Add region names
        result = pd.merge(
            region_avg.reset_index(),
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )
        
        self.logger.info(f"Created region averages with shape: {result.shape}")
        
        return result
        
    def save_results(self, 
                    cohort_prefix: str, 
                    results: Dict[str, Dict[str, pd.DataFrame]]):
        """Save aggregated results to files"""
        for dataset_type, dataset_results in results.items():
            prefix = f"ge_{cohort_prefix}" if dataset_type == 'filtered' else cohort_prefix
            
            for feature_type, df in dataset_results.items():
                output_path = os.path.join(
                    self.config['paths']['results'],
                    f"{prefix}_{feature_type}.csv"
                )
                df.to_csv(output_path, index=False)
                self.logger.info(f"Saved {dataset_type} {feature_type} to {output_path}")