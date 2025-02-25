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

        # Updated colors similar to your previous scheme
        self.colors = {
            'hup_regions': '#8db9c7',   # Pastel blue
            'mni_regions': '#536878',   # Deep teal
            'violin_alpha': 0.5
        }

    def plot_region_level_comparison(self,
                                     hup_regions: pd.DataFrame,
                                     mni_regions: pd.DataFrame,
                                     statistical_results: List[Dict],
                                     feature_type: str,
                                     output_path: str):
        """
        Plot comparison showing region-level data only.
        """
        features = self._get_features(feature_type)
        num_features = len(features)
        fig, ax = self._setup_plot(feature_type)
        positions = np.arange(num_features) * 2  # Increase spacing between features

        max_data_value = None  # To track the overall maximum y-value

        for idx, feature in enumerate(features):
            stat_result = next((r for r in statistical_results if r['feature'] == feature), None)
            if stat_result is None:
                continue

            # Positions for the two cohorts
            pos_hup = positions[idx] - 0.4
            pos_mni = positions[idx] + 0.4

            # Calculate midpoint for annotation
            midpoint = (pos_hup + pos_mni) / 2

            # Violin plots
            data_hup = hup_regions[feature].dropna()
            data_mni = mni_regions[feature].dropna()
            data = [data_hup, data_mni]
            positions_violin = [pos_hup, pos_mni]
            vp = ax.violinplot(data, positions=positions_violin, widths=0.8, showmeans=True)

            # Style the violins
            colors_violin = [
                self.colors['hup_regions'],
                self.colors['mni_regions']
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
            x_hup = np.random.normal(pos_hup, jitter, size=len(data[0]))
            x_mni = np.random.normal(pos_mni, jitter, size=len(data[1]))

            ax.scatter(x_hup, data[0],
                       color=self.colors['hup_regions'], alpha=0.6, s=20,
                       label='HUP regions' if idx == 0 else None)
            ax.scatter(x_mni, data[1],
                       color=self.colors['mni_regions'], alpha=0.6, s=20,
                       label='MNI regions' if idx == 0 else None)

            # Update the maximum data value
            current_max = max(data_hup.max(), data_mni.max())
            if max_data_value is None or current_max > max_data_value:
                max_data_value = current_max

            # Add statistical annotations at the midpoint
            self._add_statistical_annotation(ax, midpoint, stat_result, current_max)

        # Adjust y-axis limits to accommodate annotations
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.15 * y_range)  # Add 15% padding to the top

        # Customize x-axis
        xticks = positions
        xtick_labels = [self.feature_mapping[f] for f in features]

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10)  # Set labels to horizontal alignment

        # Adjust font size if labels overlap
        if num_features > 5:
            plt.setp(ax.get_xticklabels(), fontsize=8)

        ax.set_ylabel('Relative Band Power' if feature_type == 'spectral' else 'Entropy')

        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _add_statistical_annotation(self, ax, position, result: Dict, data_max: float):
        """Add statistical annotation to plot."""
        significance = self._get_significance_stars(result['p_value_fdr'])
        p_value_formatted = self._format_p_value(result['p_value_fdr'])
        text = f'{significance}\np={p_value_formatted}\nES={result["effect_size"]:.2f}'

        # Determine text position
        # Place text slightly above the maximum data point, but within y-axis limits
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin

        # Calculate potential text_y position
        text_y = data_max + 0.02 * y_range

        # Ensure text_y does not exceed ymax
        if text_y > ymax:
            text_y = ymax - 0.02 * y_range  # Slightly below ymax

        ax.text(position, text_y, text,
                ha='center', va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    def _format_p_value(self, p_value: float) -> str:
        """Format p-value to display significant digits without showing 0.000."""
        if p_value < 0.001:
            # Use scientific notation for very small p-values
            return f"{p_value:.1e}"
        else:
            # Display up to 3 decimal places
            return f"{p_value:.3f}"

    def _get_significance_stars(self, p_value: float) -> str:
        """Return a string of asterisks based on p-value thresholds."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    def _get_features(self, feature_type: str) -> List[str]:
        """Get list of features based on type."""
        return [f for f in self.feature_mapping.keys()
                if ('Rel' in f) == (feature_type == 'spectral')]

    def _setup_plot(self, feature_type: str) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plot figure and axes."""
        num_features = len(self._get_features(feature_type))
        fig_width = max(6, num_features * 2.5)  # Ensure minimum width of 6 inches
        fig, ax = plt.subplots(figsize=(fig_width, 5))  # Set height to 5 inches
        ax.set_title(f'{"Frequency-domain" if feature_type == "spectral" else "Entropy"} features between sites')
        return fig, ax
