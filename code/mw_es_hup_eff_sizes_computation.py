# mw_es_hup_eff_sizes_computation.py
import numpy as np
from scipy import stats
from typing import Dict
from statsmodels.stats.multitest import multipletests
from mw_es_hup_config import FEATURE_COLUMNS, MIN_PATIENTS

class MannWhitneyEffectSizeAnalyzer:
    def __init__(self, hup_features, mni_features, min_patients=MIN_PATIENTS):
        self.hup_features = hup_features
        self.mni_features = mni_features
        self.min_patients = min_patients
        self.results = {}

    @staticmethod
    def compute_mann_whitney_stats(group1: np.ndarray, group2: np.ndarray):
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        n1, n2 = len(group1), len(group2)
        auc = stat / (n1 * n2)
        return stat, auc, p_value

    def analyze_effect_sizes(self) -> Dict:
        common_regions = set(self.hup_features['roi'].unique()) & set(self.mni_features['roi'].unique())
        
        all_p_values = []
        region_feature_pairs = []

        for region in common_regions:
            hup_patients = self.hup_features[self.hup_features['roi'] == region]['patient_id'].nunique()
            mni_patients = self.mni_features[self.mni_features['roi'] == region]['patient_id'].nunique()

            if hup_patients < self.min_patients or mni_patients < self.min_patients:
                continue

            self.results[region] = {}
            for feature in FEATURE_COLUMNS:
                hup_data = self.hup_features[self.hup_features['roi'] == region][feature].values
                mni_data = self.mni_features[self.mni_features['roi'] == region][feature].values

                u_stat, auc, p_value = self.compute_mann_whitney_stats(hup_data, mni_data)
                all_p_values.append(p_value)
                region_feature_pairs.append((region, feature))

                self.results[region][feature] = {
                    'U_statistic': u_stat,
                    'AUC': auc,
                    'p_value': p_value,
                    'hup_n': len(hup_data),
                    'mni_n': len(mni_data)
                }

        # Multiple comparisons
        if len(all_p_values) > 0:
            self._apply_multiple_comparisons(all_p_values, region_feature_pairs)

        return self.results

    def _apply_multiple_comparisons(self, all_p_values, region_feature_pairs):
        all_p_values = np.array(all_p_values)
        bonferroni_threshold = 0.05 / len(all_p_values)
        bonferroni_significant = all_p_values < bonferroni_threshold
        _, p_values_fdr, _, _ = multipletests(all_p_values, method='fdr_bh')

        for (region, feature), p_orig, p_fdr, is_bonf_sig in zip(region_feature_pairs, all_p_values, p_values_fdr, bonferroni_significant):
            self.results[region][feature].update({
                'p_value_bonferroni_threshold': bonferroni_threshold,
                'significant_bonferroni': is_bonf_sig,
                'p_value_fdr': p_fdr,
                'significant_fdr': p_fdr < 0.05
            })
