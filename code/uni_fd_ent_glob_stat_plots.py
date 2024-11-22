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

        # Distinct colors for each group
        self.colors = {
            'hup_electrodes': '#1f77b4',   # Blue
            'hup_regions': '#17becf',      # Cyan
            'mni_electrodes': '#d62728',   # Red
            'mni_regions': '#ff7f0e',      # Orange
            'violin_alpha': 0.5
        }

    def plot_electrode_and_region_comparison(self,
                                             hup_electrodes: pd.DataFrame,
                                             mni_electrodes: pd.DataFrame,
                                             hup_regions: pd.DataFrame,
                                             mni_regions: pd.DataFrame,
                                             statistical_results: List[Dict],
                                             feature_type: str,
                                             output_path: str):
        """
        Plot comparison showing both individual electrodes and region averages
        with two pairs of violin plots per feature (total four violins per feature).
        """
        features = self._get_features(feature_type)
        num_features = len(features)
        fig, ax = plt.subplots(figsize=(num_features * 2.5, 8))
        positions = np.arange(num_features) * 5  # Increase spacing between features

        for idx, feature in enumerate(features):
            stat_result = next((r for r in statistical_results if r['feature'] == feature), None)
            if stat_result is None:
                continue

            # Positions for the four groups
            pos_hup_electrodes = positions[idx] - 1.5
            pos_hup_regions = positions[idx] - 0.5
            pos_mni_regions = positions[idx] + 0.5
            pos_mni_electrodes = positions[idx] + 1.5

            # Violin plots
            data = [
                hup_electrodes[feature].dropna(),
                hup_regions[feature].dropna(),
                mni_regions[feature].dropna(),
                mni_electrodes[feature].dropna()
            ]
            positions_violin = [pos_hup_electrodes, pos_hup_regions, pos_mni_regions, pos_mni_electrodes]
            vp = ax.violinplot(data, positions=positions_violin, widths=0.8, showmeans=True)

            # Style the violins
            colors_violin = [
                self.colors['hup_electrodes'],
                self.colors['hup_regions'],
                self.colors['mni_regions'],
                self.colors['mni_electrodes']
            ]
            for i, pc in enumerate(vp['bodies']):
                pc.set_facecolor(colors_violin[i])
                pc.set_edgecolor('black')
                pc.set_alpha(self.colors['violin_alpha'])
            for partname in ('cmeans', 'cbars', 'cmaxes', 'cmins'):
                vp[partname].set_edgecolor('black')
                vp[partname].set_linewidth(1)

            # Plot individual data points with slight horizontal jitter
            jitter = 0.08
            x_hup_electrodes = np.random.normal(pos_hup_electrodes, jitter, size=len(data[0]))
            x_hup_regions = np.random.normal(pos_hup_regions, jitter, size=len(data[1]))
            x_mni_regions = np.random.normal(pos_mni_regions, jitter, size=len(data[2]))
            x_mni_electrodes = np.random.normal(pos_mni_electrodes, jitter, size=len(data[3]))

            ax.scatter(x_hup_electrodes, data[0],
                       color=self.colors['hup_electrodes'], alpha=0.6, s=5,
                       label='HUP electrodes' if idx == 0 else None)
            ax.scatter(x_hup_regions, data[1],
                       color=self.colors['hup_regions'], alpha=0.8, s=20,
                       label='HUP regions' if idx == 0 else None)
            ax.scatter(x_mni_regions, data[2],
                       color=self.colors['mni_regions'], alpha=0.8, s=20,
                       label='MNI regions' if idx == 0 else None)
            ax.scatter(x_mni_electrodes, data[3],
                       color=self.colors['mni_electrodes'], alpha=0.6, s=5,
                       label='MNI electrodes' if idx == 0 else None)

            # Add statistical annotations
            self._add_statistical_annotation(ax, positions[idx], stat_result)

        # Customize x-axis
        xticks = []
        xtick_labels = []
        for idx, feature in enumerate(features):
            xticks.extend([
                positions[idx] - 1.5,
                positions[idx] - 0.5,
                positions[idx] + 0.5,
                positions[idx] + 1.5
            ])
            xtick_labels.extend([
                'HUP electrodes',
                'HUP regions',
                'MNI regions',
                'MNI electrodes'
            ])

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

        # Set x-axis minor ticks to indicate features
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(positions)
        ax2.set_xticklabels([self.feature_mapping[f] for f in features], rotation=0, ha='center')
        ax2.tick_params(axis='x', which='major', pad=20)

        ax.set_ylabel('Relative Band Power' if feature_type == 'spectral' else 'Entropy')

        # Create custom legend handles
        handles = []
        # HUP electrodes
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  label='HUP electrodes',
                                  markerfacecolor=self.colors['hup_electrodes'], markersize=5))
        # HUP regions
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  label='HUP regions',
                                  markerfacecolor=self.colors['hup_regions'], markersize=8))
        # MNI regions
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  label='MNI regions',
                                  markerfacecolor=self.colors['mni_regions'], markersize=8))
        # MNI electrodes
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  label='MNI electrodes',
                                  markerfacecolor=self.colors['mni_electrodes'], markersize=5))

        ax.legend(handles=handles)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _add_statistical_annotation(self, ax, position, result: Dict):
        """Add statistical annotation to plot"""
        significance = '*' if result['p_value_fdr'] < 0.05 else ''
        text = f'{significance} p={result["p_value_fdr"]:.3f}\nES={result["effect_size"]:.2f}'

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        text_y = ymax - 0.05 * y_range

        ax.text(position, text_y, text,
                ha='center', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    def _get_features(self, feature_type: str) -> List[str]:
        """Get list of features based on type"""
        return [f for f in self.feature_mapping.keys()
                if ('Rel' in f) == (feature_type == 'spectral')]

    def _setup_plot(self, feature_type: str) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plot figure and axes"""
        num_features = len(self._get_features(feature_type))
        fig, ax = plt.subplots(figsize=(num_features * 2.5, 8))
        ax.set_title(f'{"Frequency-domain" if feature_type == "spectral" else "Entropy"} features between sites')
        return fig, ax
