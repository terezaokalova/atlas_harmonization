# src/region_aggregator.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

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
        combined_df = self._prepare_combined_df(features_df, region_df, patient_map_arr)
        
        # First average within patient-region
        patient_region_avg = combined_df.groupby(['patient_id', 'roiNum']).mean()
        
        # Then average across patients for each region
        region_avg = patient_region_avg.groupby('roiNum').mean()
        
        # Add region names
        return pd.merge(
            region_avg.reset_index(),
            dk_atlas_df[['roiNum', 'roi']].drop_duplicates('roiNum'),
            on='roiNum',
            how='left'
        )
        
    def save_results(self, 
                    cohort_prefix: str, 
                    region_features: pd.DataFrame, 
                    region_averages: pd.DataFrame):
        """Save aggregated results to files"""
        region_features.to_csv(
            f"{self.config['paths']['results']}/{cohort_prefix}_region_features.csv",
            index=False
        )
        region_averages.to_csv(
            f"{self.config['paths']['results']}/{cohort_prefix}_region_averages.csv",
            index=False
        )