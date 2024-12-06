# mw_es_hup_main.py
import os
import pickle
import pandas as pd

from mw_es_hup_config import BASE_PATH_RESULTS, FEATURE_COLUMNS, FEATURE_NAME_MAPPING
from mw_es_hup_data_loading import load_hup_features, load_mni_features
from mw_es_hup_eff_sizes_computation import MannWhitneyEffectSizeAnalyzer
from mw_es_hup_visualizations import create_auc_heatmap

def main():
    # Load data
    hup_region_features = load_hup_features()
    mni_region_features = load_mni_features()

    # Verify number of unique HUP patients
    num_hup_patients = hup_region_features['patient_id'].nunique()
    print(f"Number of contributing HUP patients: {num_hup_patients}")

    # Run analysis
    analyzer = MannWhitneyEffectSizeAnalyzer(hup_region_features, mni_region_features)
    results = analyzer.analyze_effect_sizes()

    # Save results
    with open(os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Also save as CSV
    results_df = []
    for region in results:
        for feature, res in results[region].items():
            row = {'Region': region, 'Feature': feature, **res}
            results_df.append(row)
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.csv'), index=False)

    # Create a heatmap of AUC values
    create_auc_heatmap(results, FEATURE_COLUMNS, FEATURE_NAME_MAPPING)

if __name__ == "__main__":
    main()
