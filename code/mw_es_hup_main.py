# mw_es_hup_main.py
import os
import pickle
import pandas as pd
import numpy as np
import scipy.io as sio

from mw_es_hup_config import (
    BASE_PATH_RESULTS, BASE_PATH_DATA, 
    FEATURE_COLUMNS, FEATURE_NAME_MAPPING,
    DESIKAN_KILLIANY, ATLAS_LUT, MIN_PATIENTS
)
from mw_es_hup_data_loading import load_hup_features, load_mni_features, load_desikan_killiany
from mw_es_hup_eff_sizes_computation import MannWhitneyEffectSizeAnalyzer
from mw_es_hup_visualizations import create_auc_heatmap, plot_static_brain_map
from nilearn import plotting as niplot
from matplotlib import pyplot as plt

def main():
    print("Starting Mann-Whitney effect size analysis pipeline...")

    # ---------------------------
    # Step 1: Load the data
    # ---------------------------
    print("Loading HUP and MNI feature data...")
    hup_region_features = load_hup_features()
    mni_region_features = load_mni_features()
    
    print("Verifying the number of contributing HUP patients...")
    num_hup_patients = hup_region_features['patient_id'].nunique()
    print(f"Number of contributing HUP patients: {num_hup_patients}")

    # ---------------------------
    # Step 2: Run Analysis
    # ---------------------------
    print("Running Mann-Whitney U analysis...")
    analyzer = MannWhitneyEffectSizeAnalyzer(hup_region_features, mni_region_features, min_patients=MIN_PATIENTS)
    results = analyzer.analyze_effect_sizes()
    print("Analysis complete.")

    # ---------------------------
    # Step 3: Save Results
    # ---------------------------
    print("Saving results to disk...")
    pkl_path = os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Pickle results saved to: {pkl_path}")

    results_list = []
    for region in results:
        for feature, res in results[region].items():
            row = {'Region': region, 'Feature': feature, **res}
            results_list.append(row)
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"CSV results saved to: {csv_path}")

    # ---------------------------
    # Step 4: Create AUC DataFrame (auc_df) for each feature
    # ---------------------------
    print("Loading LUT to map region names to roiNum...")
    desikan_killiany_df = load_desikan_killiany()
    region_to_roiNum = desikan_killiany_df.set_index('roi')['roiNum'].to_dict()

    print("Loading HUP atlas and merging with hup_df for MNI coordinates...")
    hup_atlas_path = os.path.join(BASE_PATH_DATA, 'HUP_atlas.mat')
    hup_df_path = os.path.join(BASE_PATH_DATA, 'hup_df.csv')
    
    if not os.path.exists(hup_atlas_path):
        raise FileNotFoundError(f"HUP atlas file not found at {hup_atlas_path}")
    if not os.path.exists(hup_df_path):
        raise FileNotFoundError(f"hup_df file not found at {hup_df_path}")
    
    hup_atlas = sio.loadmat(hup_atlas_path)
    hup_df = pd.read_csv(hup_df_path)

    mni_coords = hup_atlas['mni_coords']
    if len(mni_coords.shape) > 2:
        mni_coords = mni_coords.reshape(-1, 3)
    mni_coords_df = pd.DataFrame(mni_coords, columns=['mni_x', 'mni_y', 'mni_z'])

    hup_atlas_with_roiNum = pd.concat([mni_coords_df, hup_df.reset_index(drop=True)], axis=1)
    hup_atlas_merged_path = os.path.join(BASE_PATH_DATA, 'hup_atlas_with_roiNum.csv')
    hup_atlas_with_roiNum.to_csv(hup_atlas_merged_path, index=False)
    print(f"HUP atlas with roiNum saved to: {hup_atlas_merged_path}")

    # ---------------------------
    # Step 5: Create and Save AUC Heatmap (Aggregated)
    # ---------------------------
    print("Creating and saving AUC heatmap across all features...")
    heatmap_path = os.path.join(BASE_PATH_RESULTS, 'AUC_heatmap.png')
    create_auc_heatmap(results, FEATURE_COLUMNS, FEATURE_NAME_MAPPING, output_path=heatmap_path)
    print(f"AUC heatmap saved to: {heatmap_path}")

    # ---------------------------
    # Step 6: Create a Brain Map for Each Feature
    # ---------------------------
    print("Creating and saving brain maps for each feature...")
    for feature in FEATURE_COLUMNS:
        print(f"Processing feature: {feature}")
        auc_data = []
        for region_name, feats_dict in results.items():
            roi_num = region_to_roiNum.get(region_name)
            if roi_num is not None:
                auc_value = feats_dict.get(feature, {}).get('AUC', None)
                if auc_value is not None:
                    auc_data.append({'roiNum': roi_num, 'AUC': auc_value})
        
        if len(auc_data) == 0:
            print(f"No AUC data found for feature {feature}, skipping...")
            continue

        auc_df = pd.DataFrame(auc_data)
        merged_df = pd.merge(hup_atlas_with_roiNum, auc_df, on='roiNum', how='inner')
        aggregated_df = merged_df.groupby('roiNum').agg({
            'mni_x': 'mean',
            'mni_y': 'mean',
            'mni_z': 'mean',
            'AUC': 'mean',
            'abvr': 'first'
        }).reset_index()

        brain_map_path = os.path.join(BASE_PATH_RESULTS, f'AUC_brain_map_static_{feature}.png')
        plot_static_brain_map(aggregated_df, output_path=brain_map_path, title_suffix=feature)
        print(f"Static brain map for {feature} saved to: {brain_map_path}")
        
    # # prev - auc in parentheses
    # for feature in FEATURE_COLUMNS:
    #     print(f"Processing feature: {feature}")
    #     auc_data = []
    #     for region_name, feats_dict in results.items():
    #         roi_num = region_to_roiNum.get(region_name)
    #         if roi_num is not None:
    #             auc_value = feats_dict.get(feature, {}).get('AUC', None)
    #             if auc_value is not None:
    #                 auc_data.append({'roiNum': roi_num, 'AUC': auc_value})
        
    #     if len(auc_data) == 0:
    #         print(f"No AUC data found for feature {feature}, skipping...")
    #         continue

    #     auc_df = pd.DataFrame(auc_data)

    #     # Merge AUC values with HUP atlas (MNI coordinates + roiNum)
    #     merged_df = pd.merge(hup_atlas_with_roiNum, auc_df, on='roiNum', how='inner')

    #     # Aggregate by roiNum
    #     filtered_df = merged_df[merged_df['roiNum'].isin(auc_df['roiNum'])]
    #     aggregated_df = filtered_df.groupby('roiNum').agg({
    #         'mni_x': 'mean',
    #         'mni_y': 'mean',
    #         'mni_z': 'mean',
    #         'AUC': 'mean',
    #         'abvr': 'first'
    #     }).reset_index()

    #     # Ensure correctness
    #     assert set(aggregated_df['roiNum']) == set(auc_df['roiNum']), "Mismatch in regions!"
        
    #     # Create and save a static brain map for this feature
    #     brain_map_path = os.path.join(BASE_PATH_RESULTS, f'AUC_brain_map_static_{feature}.png')
    #     plot_static_brain_map(aggregated_df, output_path=brain_map_path)
    #     print(f"Static brain map for {feature} saved to: {brain_map_path}")

    print("All feature brain maps created.")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()

# # mw_es_hup_main.py
# import os
# import pickle
# import pandas as pd
# import numpy as np
# import scipy.io as sio

# from mw_es_hup_config import (
#     BASE_PATH_RESULTS, BASE_PATH_DATA, 
#     FEATURE_COLUMNS, FEATURE_NAME_MAPPING,
#     DESIKAN_KILLIANY, ATLAS_LUT
# )
# from mw_es_hup_data_loading import load_hup_features, load_mni_features, load_desikan_killiany
# from mw_es_hup_eff_sizes_computation import MannWhitneyEffectSizeAnalyzer
# from mw_es_hup_visualizations import create_auc_heatmap, plot_static_brain_map

# from nilearn import plotting as niplot
# from matplotlib import pyplot as plt

# def main():
#     print("Starting Mann-Whitney effect size analysis pipeline...")

#     # ---------------------------
#     # Step 1: Load the data
#     # ---------------------------
#     print("Loading HUP and MNI feature data...")
#     hup_region_features = load_hup_features()
#     mni_region_features = load_mni_features()
    
#     print("Verifying the number of contributing HUP patients...")
#     num_hup_patients = hup_region_features['patient_id'].nunique()
#     print(f"Number of contributing HUP patients: {num_hup_patients}")

#     # ---------------------------
#     # Step 2: Run Analysis
#     # ---------------------------
#     print("Running Mann-Whitney U analysis...")
#     analyzer = MannWhitneyEffectSizeAnalyzer(hup_region_features, mni_region_features)
#     results = analyzer.analyze_effect_sizes()
#     print("Analysis complete.")

#     # ---------------------------
#     # Step 3: Save Results
#     # ---------------------------
#     print("Saving results to disk...")
#     pkl_path = os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.pkl')
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(results, f)
#     print(f"Pickle results saved to: {pkl_path}")

#     results_list = []
#     for region in results:
#         for feature, res in results[region].items():
#             row = {'Region': region, 'Feature': feature, **res}
#             results_list.append(row)
#     results_df = pd.DataFrame(results_list)
#     csv_path = os.path.join(BASE_PATH_RESULTS, 'mann_whitney_results.csv')
#     results_df.to_csv(csv_path, index=False)
#     print(f"CSV results saved to: {csv_path}")

#     # ---------------------------
#     # Step 4: Create AUC DataFrame (auc_df)
#     # ---------------------------
#     print("Creating AUC DataFrame (auc_df) from results...")
#     # Load LUT to map region names to roiNum
#     desikan_killiany_df = load_desikan_killiany()
#     region_to_roiNum = desikan_killiany_df.set_index('roi')['roiNum'].to_dict()

#     auc_data = []
#     for region_name, features in results.items():
#         roi_num = region_to_roiNum.get(region_name)
#         if roi_num is not None:
#             # Choose a feature for example, or integrate all.
#             # Here, we just need one for demonstration, but you can choose a representative feature.
#             # To keep consistent with previous code, let's pick 'deltaRel_mean' as representative:
#             auc_value = features.get('deltaRel_mean', {}).get('AUC', None)
#             if auc_value is not None:
#                 auc_data.append({'roiNum': roi_num, 'AUC': auc_value})
#     auc_df = pd.DataFrame(auc_data)
#     print("AUC DataFrame created.")

#     # ---------------------------
#     # Step 5: Load HUP atlas and prepare merged_df
#     # ---------------------------
#     # We assume you have HUP_atlas.mat and hup_df.csv in Data
#     print("Loading HUP atlas and merging with hup_df for MNI coordinates...")
#     hup_atlas_path = os.path.join(BASE_PATH_DATA, 'HUP_atlas.mat')
#     hup_df_path = os.path.join(BASE_PATH_DATA, 'hup_df.csv')
    
#     if not os.path.exists(hup_atlas_path):
#         raise FileNotFoundError(f"HUP atlas file not found at {hup_atlas_path}")
#     if not os.path.exists(hup_df_path):
#         raise FileNotFoundError(f"hup_df file not found at {hup_df_path}")
    
#     hup_atlas = sio.loadmat(hup_atlas_path)
#     hup_df = pd.read_csv(hup_df_path)

#     mni_coords = hup_atlas['mni_coords']
#     if len(mni_coords.shape) > 2:
#         mni_coords = mni_coords.reshape(-1, 3)
#     mni_coords_df = pd.DataFrame(mni_coords, columns=['mni_x', 'mni_y', 'mni_z'])

#     hup_atlas_with_roiNum = pd.concat([mni_coords_df, hup_df.reset_index(drop=True)], axis=1)
#     # Save for inspection if needed
#     hup_atlas_merged_path = os.path.join(BASE_PATH_DATA, 'hup_atlas_with_roiNum.csv')
#     hup_atlas_with_roiNum.to_csv(hup_atlas_merged_path, index=False)
#     print(f"HUP atlas with roiNum saved to: {hup_atlas_merged_path}")

#     # Merge AUC values with HUP atlas (MNI coordinates + roiNum)
#     print("Merging AUC values with HUP atlas info to create merged_df...")
#     merged_df = pd.merge(hup_atlas_with_roiNum, auc_df, on='roiNum', how='inner')
#     print("merged_df created.")

#     # ---------------------------
#     # Step 6: Aggregate Data
#     # ---------------------------
#     print("Aggregating AUC values and coordinates by region...")
#     filtered_df = merged_df[merged_df['roiNum'].isin(auc_df['roiNum'])]
#     aggregated_df = filtered_df.groupby('roiNum').agg({
#         'mni_x': 'mean',
#         'mni_y': 'mean',
#         'mni_z': 'mean',
#         'AUC': 'mean',
#         'abvr': 'first'
#     }).reset_index()

#     # Verification
#     print(f"Number of regions in aggregated_df: {aggregated_df['roiNum'].nunique()}")
#     print(f"Regions in aggregated_df: {aggregated_df['roiNum'].tolist()}")
#     print(f"Regions in auc_df: {auc_df['roiNum'].tolist()}")
#     assert set(aggregated_df['roiNum']) == set(auc_df['roiNum']), "Mismatch in regions!"
#     print("Aggregation complete and verified.")

#     # ---------------------------
#     # Step 7: Create and Save Visualizations
#     # ---------------------------
#     print("Creating and saving AUC heatmap...")
#     heatmap_path = os.path.join(BASE_PATH_RESULTS, 'AUC_heatmap.png')
#     create_auc_heatmap(results, FEATURE_COLUMNS, FEATURE_NAME_MAPPING, output_path=heatmap_path)
#     print(f"AUC heatmap saved to: {heatmap_path}")

#     print("Creating and saving static brain map visualization...")
#     brain_map_path = os.path.join(BASE_PATH_RESULTS, 'AUC_brain_map_static.png')
#     plot_static_brain_map(aggregated_df, output_path=brain_map_path)
#     print(f"Static brain map saved to: {brain_map_path}")

#     print("Pipeline complete.")

# if __name__ == "__main__":
#     main()
