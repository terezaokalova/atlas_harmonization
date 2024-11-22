# uni_fd_ent_glob_stat_plots.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class UnivariateVisualization:
    def __init__(self):
        self.feature_mapping = {
            'deltaRel': 'delta (0.5–4 Hz)',
            'thetaRel': 'theta (4–8 Hz)',
            'alphaRel': 'alpha (8–13 Hz)',
            'betaRel': 'beta (13–30 Hz)',
            'gammaRel': 'gamma (30–80 Hz)',
            'entropy_1min': 'entropy (1-min segment)',
            'entropy_fullts': 'entropy (full time series)'
        }
        
        self.colors = {
            'ge_go_hup': '#8db9c7',      # pastel blue
            'mni': '#536878',            # deep teal
            'ge_go_hup_avg': '#5d89a8',  # darker blue for region averages
            'mni_avg': '#374f6b',        # darker teal for region averages
            'violin_alpha': 0.2
        }
    
    def plot_region_level_comparison(self, 
                                     ge_go_hup_regions: pd.DataFrame,
                                     mni_regions: pd.DataFrame,
                                     statistical_results: List[Dict],
                                     feature_type: str,
                                     output_path: str):
        """
        Plot comparison showing individual regions
        """
        features = self._get_features(feature_type)
        fig, ax = self._setup_plot(feature_type)
        positions = np.arange(len(features)) * 2 + 1
        
        for idx, feature in enumerate(features):
            stat_result = next(r for r in statistical_results if r['feature'] == feature)
            
            # Plot individual regions
            ax.scatter([positions[idx] - 0.5] * len(ge_go_hup_regions),
                       ge_go_hup_regions[feature].values,
                       color=self.colors['ge_go_hup'],
                       alpha=0.6, s=30,
                       label='GE-GO HUP regions' if idx == 0 else None)
            
            ax.scatter([positions[idx] + 0.5] * len(mni_regions),
                       mni_regions[feature].values,
                       color=self.colors['mni'],
                       alpha=0.6, s=30,
                       label='MNI regions' if idx == 0 else None)
            
            # Add violin plots for distribution
            vp_data = [ge_go_hup_regions[feature].values,
                       mni_regions[feature].values]
            vp = ax.violinplot(vp_data,
                               positions=[positions[idx] - 0.5, positions[idx] + 0.5],
                               showmeans=True)
            
            self._style_violins(vp)
            self._add_statistical_annotation(ax, positions[idx], stat_result)
        
        self._finalize_plot(ax, positions, features, feature_type,
                            f'GE-GO HUP: {len(ge_go_hup_regions)} regions\n'
                            f'MNI: {len(mni_regions)} regions')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_electrode_and_region_comparison(self,
                                             ge_go_hup_electrodes: pd.DataFrame,
                                             mni_electrodes: pd.DataFrame,
                                             ge_go_hup_regions: pd.DataFrame,
                                             mni_regions: pd.DataFrame,
                                             statistical_results: List[Dict],
                                             feature_type: str,
                                             output_path: str):
        """
        Plot comparison showing both individual electrodes and region averages
        """
        features = self._get_features(feature_type)
        fig, ax = self._setup_plot(feature_type)
        positions = np.arange(len(features)) * 2 + 1
        
        for idx, feature in enumerate(features):
            stat_result = next(r for r in statistical_results if r['feature'] == feature)
            
            # Plot individual electrodes with low alpha
            ax.scatter([positions[idx] - 0.5] * len(ge_go_hup_electrodes),
                       ge_go_hup_electrodes[feature].values,
                       color=self.colors['ge_go_hup'],
                       alpha=0.1, s=5,
                       label='GE-GO HUP electrodes' if idx == 0 else None)
            
            ax.scatter([positions[idx] + 0.5] * len(mni_electrodes),
                       mni_electrodes[feature].values,
                       color=self.colors['mni'],
                       alpha=0.1, s=5,
                       label='MNI electrodes' if idx == 0 else None)
            
            # Plot region averages with larger markers
            ax.scatter([positions[idx] - 0.5] * len(ge_go_hup_regions),
                       ge_go_hup_regions[feature].values,
                       color=self.colors['ge_go_hup_avg'],
                       s=50, marker='*',
                       label='GE-GO HUP region averages' if idx == 0 else None)
            
            ax.scatter([positions[idx] + 0.5] * len(mni_regions),
                       mni_regions[feature].values,
                       color=self.colors['mni_avg'],
                       s=50, marker='*',
                       label='MNI region averages' if idx == 0 else None)
            
            # Add violin plots
            vp_data = [ge_go_hup_electrodes[feature].values,
                       mni_electrodes[feature].values]
            vp = ax.violinplot(vp_data,
                               positions=[positions[idx] - 0.5, positions[idx] + 0.5],
                               showmeans=False, showextrema=False)
            
            self._style_violins(vp)
            self._add_statistical_annotation(ax, positions[idx], stat_result)
        
        self._finalize_plot(ax, positions, features, feature_type,
                            f'GE-GO HUP: {len(ge_go_hup_regions)} regions, {len(ge_go_hup_electrodes)} electrodes\n'
                            f'MNI: {len(mni_regions)} regions, {len(mni_electrodes)} electrodes')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_features(self, feature_type: str) -> List[str]:
        """Get list of features based on type"""
        return [f for f in self.feature_mapping.keys() 
                if ('Rel' in f) == (feature_type == 'spectral')]
    
    def _setup_plot(self, feature_type: str) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plot figure and axes"""
        fig, ax = plt.subplots(figsize=(12, 8) if feature_type == 'spectral' else (10, 5))
        ax.set_title(f'{"Frequency-domain" if feature_type == "spectral" else "Entropy"} '
                     f'features between sites')
        return fig, ax
    
    def _style_violins(self, vp):
        """Style violin plots"""
        for idx, pc in enumerate(vp['bodies']):
            pc.set_facecolor(self.colors['ge_go_hup'] if idx == 0 else self.colors['mni'])
            pc.set_alpha(self.colors['violin_alpha'])
    
    def _add_statistical_annotation(self, ax, position, result: Dict):
        """Add statistical annotation to plot"""
        significance = '*' if result['p_value_fdr'] < 0.05 else ''
        text = f'{significance}\np={result["p_value_fdr"]:.3f}\nES={result["effect_size"]:.2f}'
        
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        text_y = min(ymax - 0.05 * y_range, 
                     max(result['hup_mean'], result['mni_mean']) + 0.1 * y_range)
        
        ax.text(position, text_y, text,
                ha='center', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    def _finalize_plot(self, ax, positions, features, feature_type, legend_title):
        """Finalize plot settings"""
        ax.set_xticks(positions)
        ax.set_xticklabels([self.feature_mapping[f] for f in features])
        ax.set_ylabel('Relative Band Power' if feature_type == 'spectral' else 'Entropy')
        ax.legend(title=legend_title)
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
        plt.tight_layout()
