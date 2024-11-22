# uni_fd_ent_stat.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

class StatisticalAnalyzer:
    def __init__(self):
        pass

    def compare_sites_globally_avg(self, hup_avg, mni_avg):
        feature_order = [
            'deltaRel',    # 0.5-4 Hz
            'thetaRel',    # 4-8 Hz
            'alphaRel',    # 8-13 Hz
            'betaRel',     # 13-30 Hz
            'gammaRel',    # 30-80 Hz
            'entropy_1min',
            'entropy_fullts'
        ]
        results = []
        for feature in feature_order:
            if feature not in hup_avg.columns:
                continue
            hup_values = hup_avg[feature].dropna()
            mni_values = mni_avg[feature].dropna()
            # Use Mann-Whitney U test for all features
            stat, p_val = stats.mannwhitneyu(hup_values, mni_values, alternative='two-sided')
            effect_size = self.calculate_effect_size(hup_values, mni_values)
            results.append({
                'feature': feature,
                'test': 'Mann-Whitney U',
                'p_value': p_val,
                'effect_size': effect_size,
                'hup_mean': np.mean(hup_values),
                'mni_mean': np.mean(mni_values),
                'hup_std': np.std(hup_values),
                'mni_std': np.std(mni_values)
            })
        results_df = pd.DataFrame(results)
        # Apply FDR correction
        reject, pvals_corrected = fdrcorrection(results_df['p_value'])
        results_df['p_value_fdr'] = pvals_corrected
        results_df['significant'] = reject
        return results_df

    def calculate_effect_size(self, group1, group2):
        # Calculate Cohen's d for effect size
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (np.mean(group1) - np.mean(group2)) / s_pooled
        return d

    def interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size"""
        effect_size = abs(effect_size)
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

    def interpret_significance(self, p_value):
        """Interpret statistical significance levels"""
        if p_value < 0.001:
            return "highly significant"
        elif p_value < 0.01:
            return "very significant"
        elif p_value < 0.05:
            return "significant"
        else:
            return "not significant"
