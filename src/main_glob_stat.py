# main_glob_stat.py

import pandas as pd
import os
from uni_fd_ent_stat import StatisticalAnalyzer
from uni_fd_ent_glob_stat_plots import UnivariateVisualization

def print_results(results_df):
    print("Global comparison results (region-averaged)")
    print("==========================================")

    significant_results = results_df[results_df['p_value_fdr'] < 0.05]
    non_significant_results = results_df[results_df['p_value_fdr'] >= 0.05]

    if not significant_results.empty:
        print("\nSignificant findings:")
        for _, row in significant_results.iterrows():
            effect_size_desc = StatisticalAnalyzer().interpret_effect_size(row['effect_size'])
            print(f"\n{row['feature']}:")
            print(f"- Significance: {row['p_value_fdr']:.3e} (FDR-corrected p-value)")
            print(f"- Effect size: {row['effect_size']:.3f} ({effect_size_desc})")
            direction = "HUP higher than MNI" if row['effect_size'] > 0 else "HUP lower than MNI"
            print(f"- Direction: {direction}")
            print(f"- Means: HUP={row['hup_mean']:.3f}±{row['hup_std']:.3f}, MNI={row['mni_mean']:.3f}±{row['mni_std']:.3f}")

    if not non_significant_results.empty:
        print("\nNon-significant findings:")
        for _, row in non_significant_results.iterrows():
            effect_size_desc = StatisticalAnalyzer().interpret_effect_size(row['effect_size'])
            print(f"\n{row['feature']}:")
            print(f"- Significance: {row['p_value_fdr']:.3e} (FDR-corrected p-value)")
            print(f"- Effect size: {row['effect_size']:.3f} ({effect_size_desc})")
            direction = "HUP higher than MNI" if row['effect_size'] > 0 else "HUP lower than MNI"
            print(f"- Direction: {direction}")
            print(f"- Means: HUP={row['hup_mean']:.3f}±{row['hup_std']:.3f}, MNI={row['mni_mean']:.3f}±{row['mni_std']:.3f}")

def main():
    # Define paths
    base_path_results = '../results'
    figures_path = '../figures'
    os.makedirs(figures_path, exist_ok=True)

    # Load data
    hup_regions = pd.read_csv(f'{base_path_results}/ge_go_hup_region_averages.csv')
    mni_regions = pd.read_csv(f'{base_path_results}/mni_region_averages.csv')

    # Load electrode-level data to get counts (even though we focus on region-level analysis)
    hup_electrodes = pd.read_csv(f'{base_path_results}/ge_go_hup_electrode_features.csv')
    mni_electrodes = pd.read_csv(f'{base_path_results}/mni_electrode_features.csv')

    # Print information about input datasets
    print("Input Datasets:")
    print("----------------")
    print(f"HUP Region Averages Dataset: {base_path_results}/ge_go_hup_region_averages.csv")
    print(f"MNI Region Averages Dataset: {base_path_results}/mni_region_averages.csv")
    print(f"HUP Electrode Features Dataset: {base_path_results}/ge_go_hup_electrode_features.csv")
    print(f"MNI Electrode Features Dataset: {base_path_results}/mni_electrode_features.csv\n")

    # Verify and print the number of electrodes from HUP and MNI
    num_hup_electrodes = len(hup_electrodes)
    num_mni_electrodes = len(mni_electrodes)
    print(f"Number of electrodes from HUP: {num_hup_electrodes}")
    print(f"Number of electrodes from MNI: {num_mni_electrodes}\n")

    # Number of regions in HUP and MNI datasets
    num_hup_regions = len(hup_regions)
    num_mni_regions = len(mni_regions)
    print(f"Number of regions in HUP dataset: {num_hup_regions}")
    print(f"Number of regions in MNI dataset: {num_mni_regions}\n")

    # Perform statistical analysis
    analyzer = StatisticalAnalyzer()
    results_df = analyzer.compare_sites_globally_avg(hup_regions, mni_regions)

    # Save results
    results_df.to_csv(f'{base_path_results}/statistical_results.csv', index=False)

    # Print results
    print_results(results_df)

    # Generate plots
    viz = UnivariateVisualization()

    # Spectral features
    viz.plot_region_level_comparison(
        hup_regions=hup_regions,
        mni_regions=mni_regions,
        statistical_results=results_df.to_dict('records'),
        feature_type='spectral',
        output_path=f'{figures_path}/spectral_features_region_level.png'
    )

    # Entropy features
    viz.plot_region_level_comparison(
        hup_regions=hup_regions,
        mni_regions=mni_regions,
        statistical_results=results_df.to_dict('records'),
        feature_type='entropy',
        output_path=f'{figures_path}/entropy_features_region_level.png'
    )

    print("\nAnalysis and plotting completed successfully.")

if __name__ == '__main__':
    main()
