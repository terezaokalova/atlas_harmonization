# mw_es_hup_visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nilearn import plotting as niplot
from matplotlib import cm, colors
from typing import Dict

def create_auc_heatmap(results: Dict, feature_order, feature_name_mapping, output_path=None):
    df = _results_to_dataframe(results, include_corrections=False)

    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot(index='Region', columns='Feature', values='AUC')
    heatmap_data = heatmap_data[feature_order]
    heatmap_data.columns = [feature_name_mapping[col] for col in heatmap_data.columns]

    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0.5,
                vmin=0, vmax=1, annot=True, fmt='.2f')
    plt.title('AUC Values (Mann-Whitney U Test)', ha='center')
    plt.xticks(rotation=0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_static_brain_map(aggregated_df, output_path=None):
    coords = aggregated_df[['mni_x', 'mni_y', 'mni_z']].values
    auc_values = aggregated_df['AUC'].values

    niplot.plot_markers(
        node_values=auc_values,
        node_coords=coords,
        node_cmap='viridis',
        node_vmin=auc_values.min(),
        node_vmax=auc_values.max(),
        node_size=30
    )

    plt.suptitle('HUP Effect Sizes (AUC)', x=0.5, ha='center')
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

def view_interactive_brain_map(aggregated_df):
    coords = aggregated_df[['mni_x', 'mni_y', 'mni_z']].values
    auc_values = aggregated_df['AUC'].values

    norm = colors.Normalize(vmin=auc_values.min(), vmax=auc_values.max())
    cmap = cm.get_cmap('viridis')
    marker_colors = cmap(norm(auc_values))

    view = niplot.view_markers(
        marker_coords=coords,
        marker_color=marker_colors,
        marker_size=8,
        marker_labels=aggregated_df['abvr'].tolist()
    )
    return view

def _results_to_dataframe(results: Dict, include_corrections=True) -> pd.DataFrame:
    df_list = []
    for region, feats in results.items():
        for feature, res in feats.items():
            row = {
                'Region': region,
                'Feature': feature,
                'AUC': res['AUC'],
                'p_value': res['p_value']
            }
            if include_corrections:
                row.update({
                    'p_value_fdr': res.get('p_value_fdr', np.nan),
                    'significant_bonferroni': res.get('significant_bonferroni', False),
                    'significant_fdr': res.get('significant_fdr', False)
                })
            df_list.append(row)
    return pd.DataFrame(df_list)
