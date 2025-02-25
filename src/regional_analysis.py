# regional_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import logging
import os
from typing import Dict, List, Tuple

# Global configurations
code_directory = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/code'
os.chdir(code_directory)

RESULTS_DIR = '../results'
DATA_DIR = '../Data'
FIGURES_DIR = '../figures'

# handles class imbalance by truncating

class RegionalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_columns = [
            'deltaRel_mean', 
            'thetaRel_mean', 
            'alphaRel_mean', 
            'betaRel_mean',
            'gammaRel_mean',
            'entropy_1min_mean',
            'entropy_fullts_mean'
        ]
        
        self.feature_names = {
            'deltaRel_mean': 'Delta Band (0.5-4 Hz)',
            'thetaRel_mean': 'Theta Band (4-8 Hz)',
            'alphaRel_mean': 'Alpha Band (8-13 Hz)',
            'betaRel_mean': 'Beta Band (13-30 Hz)',
            'gammaRel_mean': 'Gamma Band (30-80 Hz)',
            'entropy_1min_mean': 'Signal Entropy (1-min)',
            'entropy_fullts_mean': 'Signal Entropy (full)'
        }
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare the regional feature data for both cohorts
        """
        try:
            # Load data with correct paths
            hup_path = os.path.join(RESULTS_DIR, 'ge_go_hup_region_features.csv')
            mni_path = os.path.join(RESULTS_DIR, 'mni_region_features.csv')
            
            print(f"Loading HUP features from: {hup_path}")
            print(f"Loading MNI features from: {mni_path}")
            
            self.hup_features = pd.read_csv(hup_path)
            self.mni_features = pd.read_csv(mni_path)
            
            # Get common regions
            self.common_regions = set(self.hup_features['roi'].unique()) & set(self.mni_features['roi'].unique())
            
            print(f"\nLoaded data:")
            print(f"HUP features shape: {self.hup_features.shape}")
            print(f"MNI features shape: {self.mni_features.shape}")
            print(f"Number of common regions: {len(self.common_regions)}")
            
            return self.hup_features, self.mni_features
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            print(f"\nCurrent working directory: {os.getcwd()}")
            print(f"RESULTS_DIR path: {os.path.abspath(RESULTS_DIR)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_paired_data(self, region: str, feature: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare paired data for a specific region and feature.
        Returns arrays of equal length for valid statistical comparison.
        """
        # Get data for specific region
        hup_region_data = self.hup_features[self.hup_features['roi'] == region]
        mni_region_data = self.mni_features[self.mni_features['roi'] == region]
        
        # Group by patient and get mean values (in case of multiple electrodes per region per patient)
        hup_data = hup_region_data.groupby('patient_id')[feature].mean().values
        mni_data = mni_region_data.groupby('patient_id')[feature].mean().values
        
        # Get minimum length to ensure paired data
        min_length = min(len(hup_data), len(mni_data))
        
        if min_length < 5:  # Minimum sample size requirement
            raise ValueError(f"Insufficient samples for region {region} (HUP: {len(hup_data)}, MNI: {len(mni_data)})")
            
        # Match lengths for pairing
        hup_data = hup_data[:min_length]
        mni_data = mni_data[:min_length]
        
        return hup_data, mni_data
    
    def compute_effect_size(self, hup_data: np.ndarray, mni_data: np.ndarray) -> float:
        """
        Compute Cohen's d effect size for paired data
        """
        diff = hup_data - mni_data
        d = np.mean(diff) / np.std(diff)
        return d
    
    def analyze_regions(self) -> pd.DataFrame:
        """
        Perform paired regional analysis between HUP and MNI cohorts
        """
        results = []
        
        for region in self.common_regions:
            self.logger.info(f"Analyzing region: {region}")
            
            for feature in self.feature_columns:
                try:
                    # Get paired data
                    hup_data, mni_data = self.prepare_paired_data(region, feature)
                    
                    # Perform Wilcoxon signed-rank test (paired test)
                    statistic, pvalue = stats.wilcoxon(hup_data, mni_data)
                    
                    # Compute effect size
                    effect_size = self.compute_effect_size(hup_data, mni_data)
                    
                    results.append({
                        'region': region,
                        'feature': feature,
                        'statistic': statistic,
                        'pvalue': pvalue,
                        'effect_size': effect_size,
                        'hup_mean': np.mean(hup_data),
                        'mni_mean': np.mean(mni_data),
                        'hup_std': np.std(hup_data),
                        'mni_std': np.std(mni_data),
                        'n_samples': len(hup_data)
                    })
                    
                except ValueError as e:
                    self.logger.warning(f"Skipping {region}-{feature}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error processing {region}-{feature}: {str(e)}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Apply FDR correction
            _, fdr_pvals = fdrcorrection(results_df['pvalue'])
            results_df['pvalue_fdr'] = fdr_pvals
        else:
            self.logger.warning("No results generated from analysis")
            
        return results_df
    
    def summarize_results(self, results_df: pd.DataFrame):
        """
        Print summary of significant findings
        """
        if len(results_df) == 0:
            print("No results to summarize")
            return
            
        print("\nRegional Analysis Summary")
        print("=" * 50)
        
        for feature in self.feature_columns:
            feature_results = results_df[results_df['feature'] == feature]
            sig_results = feature_results[feature_results['pvalue_fdr'] < 0.05]
            
            print(f"\n{self.feature_names[feature]}:")
            print(f"- {len(sig_results)} regions show significant differences")
            
            if len(sig_results) > 0:
                sig_results = sig_results.sort_values('effect_size', key=abs, ascending=False)
                print("\nTop regions with largest differences:")
                
                for _, row in sig_results.head(3).iterrows():
                    direction = "higher in HUP" if row['effect_size'] > 0 else "higher in MNI"
                    effect_mag = "large" if abs(row['effect_size']) > 0.8 else \
                                "medium" if abs(row['effect_size']) > 0.5 else "small"
                    
                    print(f"  * {row['region']}: {effect_mag} effect {direction}")
                    print(f"    (p={row['pvalue_fdr']:.3e}, d={row['effect_size']:.2f}, n={row['n_samples']})")


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize analysis
        analysis = RegionalAnalysis()
        
        # Load data
        analysis.load_data()
        
        # Perform analysis
        results = analysis.analyze_regions()
        
        # Summarize results
        analysis.summarize_results(results)
        
       
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()


class RegionalAnalysis_unpaired:
    def __init__(self, hup_path: str = '../results/ge_go_hup_region_features.csv',
                 mni_path: str = '../results/mni_region_features.csv'):
        """
        Initialize the RegionalAnalysis class.
        
        Parameters:
        -----------
        hup_path : str
            Path to the HUP cohort data CSV file
        mni_path : str
            Path to the MNI cohort data CSV file
        """
        self.logger = logging.getLogger(__name__)
        self.hup_path = hup_path
        self.mni_path = mni_path
        
        # Initialize data attributes
        self.hup_features = None
        self.mni_features = None
        self.common_regions = None
        
        self.feature_columns = [
            'deltaRel_mean', 
            'thetaRel_mean', 
            'alphaRel_mean', 
            'betaRel_mean',
            'gammaRel_mean',
            'entropy_1min_mean',
            'entropy_fullts_mean'
        ]
        
        self.feature_names = {
            'deltaRel_mean': 'Delta Band (0.5-4 Hz)',
            'thetaRel_mean': 'Theta Band (4-8 Hz)',
            'alphaRel_mean': 'Alpha Band (8-13 Hz)',
            'betaRel_mean': 'Beta Band (13-30 Hz)',
            'gammaRel_mean': 'Gamma Band (30-80 Hz)',
            'entropy_1min_mean': 'Signal Entropy (1-min)',
            'entropy_fullts_mean': 'Signal Entropy (full)'
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare the regional feature data for both cohorts
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            HUP features DataFrame and MNI features DataFrame
        """
        try:
            print(f"Loading HUP features from: {self.hup_path}")
            print(f"Loading MNI features from: {self.mni_path}")
            
            self.hup_features = pd.read_csv(self.hup_path)
            self.mni_features = pd.read_csv(self.mni_path)
            
            # Get common regions
            self.common_regions = set(self.hup_features['roi'].unique()) & set(self.mni_features['roi'].unique())
            
            print(f"\nLoaded data:")
            print(f"HUP features shape: {self.hup_features.shape}")
            print(f"MNI features shape: {self.mni_features.shape}")
            print(f"Number of common regions: {len(self.common_regions)}")
            
            return self.hup_features, self.mni_features
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            print(f"\nCurrent working directory: {os.getcwd()}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_unpaired_data(self, region: str, feature: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare unpaired data for a specific region and feature.
        Returns full arrays for both cohorts without truncation.
        
        Parameters:
        -----------
        region : str
            Brain region to analyze
        feature : str
            Feature to analyze
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays containing HUP and MNI data for the specified region and feature
        """
        if self.hup_features is None or self.mni_features is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Get data for specific region
        hup_region_data = self.hup_features[self.hup_features['roi'] == region]
        mni_region_data = self.mni_features[self.mni_features['roi'] == region]
        
        # Group by patient and get mean values
        hup_data = hup_region_data.groupby('patient_id')[feature].mean().values
        mni_data = mni_region_data.groupby('patient_id')[feature].mean().values
        
        if len(hup_data) < 5 or len(mni_data) < 5:
            raise ValueError(f"Insufficient samples for region {region} (HUP: {len(hup_data)}, MNI: {len(mni_data)})")
            
        return hup_data, mni_data
    
    def compute_effect_size(self, hup_data: np.ndarray, mni_data: np.ndarray) -> float:
        """
        Compute Cohen's d effect size for unpaired data
        
        Parameters:
        -----------
        hup_data : np.ndarray
            Array of HUP cohort data
        mni_data : np.ndarray
            Array of MNI cohort data
            
        Returns:
        --------
        float
            Cohen's d effect size
        """
        n1, n2 = len(hup_data), len(mni_data)
        var1, var2 = np.var(hup_data, ddof=1), np.var(mni_data, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(hup_data) - np.mean(mni_data)) / pooled_sd
        return d
    
    def analyze_regions(self) -> pd.DataFrame:
        """
        Perform unpaired regional analysis between HUP and MNI cohorts
        
        Returns:
        --------
        pd.DataFrame
            Results DataFrame containing statistical analysis results
        """
        if self.common_regions is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        results = []
        
        for region in self.common_regions:
            self.logger.info(f"Analyzing region: {region}")
            
            for feature in self.feature_columns:
                try:
                    # Get unpaired data
                    hup_data, mni_data = self.prepare_unpaired_data(region, feature)
                    
                    # Check normality
                    _, hup_norm_p = stats.shapiro(hup_data)
                    _, mni_norm_p = stats.shapiro(mni_data)
                    is_normal = (hup_norm_p > 0.05) and (mni_norm_p > 0.05)
                    
                    # Check homogeneity of variance
                    _, var_p = stats.levene(hup_data, mni_data)
                    equal_var = var_p > 0.05
                    
                    # Choose appropriate statistical test
                    if is_normal and equal_var:
                        # Use Student's t-test
                        statistic, pvalue = stats.ttest_ind(hup_data, mni_data)
                        test_used = "t-test"
                    elif is_normal and not equal_var:
                        # Use Welch's t-test
                        statistic, pvalue = stats.ttest_ind(hup_data, mni_data, equal_var=False)
                        test_used = "Welch"
                    else:
                        # Use Mann-Whitney U test
                        statistic, pvalue = stats.mannwhitneyu(hup_data, mni_data, alternative='two-sided')
                        test_used = "Mann-Whitney"
                    
                    # Compute effect size
                    effect_size = self.compute_effect_size(hup_data, mni_data)
                    
                    results.append({
                        'region': region,
                        'feature': feature,
                        'test_used': test_used,
                        'statistic': statistic,
                        'pvalue': pvalue,
                        'effect_size': effect_size,
                        'hup_mean': np.mean(hup_data),
                        'mni_mean': np.mean(mni_data),
                        'hup_std': np.std(hup_data),
                        'mni_std': np.std(mni_data),
                        'hup_n': len(hup_data),
                        'mni_n': len(mni_data)
                    })
                    
                except ValueError as e:
                    self.logger.warning(f"Skipping {region}-{feature}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error processing {region}-{feature}: {str(e)}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Apply FDR correction
            _, fdr_pvals = fdrcorrection(results_df['pvalue'])
            results_df['pvalue_fdr'] = fdr_pvals
        
        return results_df

    def summarize_results(self, results_df: pd.DataFrame):
        """
        Print summary of significant findings
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame containing analysis results
        """
        if len(results_df) == 0:
            print("No results to summarize")
            return
            
        print("\nRegional Analysis Summary")
        print("=" * 50)
        
        for feature in self.feature_columns:
            feature_results = results_df[results_df['feature'] == feature]
            sig_results = feature_results[feature_results['pvalue_fdr'] < 0.05]
            
            print(f"\n{self.feature_names[feature]}:")
            print(f"- {len(sig_results)} regions show significant differences")
            
            if len(sig_results) > 0:
                sig_results = sig_results.sort_values('effect_size', key=abs, ascending=False)
                print("\nTop regions with largest differences:")
                
                for _, row in sig_results.head(3).iterrows():
                    direction = "higher in HUP" if row['effect_size'] > 0 else "higher in MNI"
                    effect_mag = "large" if abs(row['effect_size']) > 0.8 else \
                                "medium" if abs(row['effect_size']) > 0.5 else "small"
                    
                    print(f"  * {row['region']}: {effect_mag} effect {direction}")
                    print(f"    (p={row['pvalue_fdr']:.3e}, d={row['effect_size']:.2f}, ")
                    print(f"     HUP n={row['hup_n']}, MNI n={row['mni_n']}, test={row['test_used']})")


def main_unpaired():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize analysis
        analysis = RegionalAnalysis_unpaired()
        
        # Load data
        analysis.load_data()
        
        # Perform analysis
        results = analysis.analyze_regions()
        
        # Summarize results
        analysis.summarize_results(results)
        
       
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main_unpaired()


