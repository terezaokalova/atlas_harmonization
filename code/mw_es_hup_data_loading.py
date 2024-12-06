# mw_es_hup_data_loading.py
import os
import pandas as pd
from mw_es_hup_config import BASE_PATH_RESULTS, BASE_PATH_DATA

def load_hup_features():
    hup_path = os.path.join(BASE_PATH_RESULTS, 'ge_go_hup_region_features.csv')
    if not os.path.exists(hup_path):
        raise FileNotFoundError(f"HUP features file not found at {hup_path}")
    return pd.read_csv(hup_path)

def load_mni_features():
    mni_path = os.path.join(BASE_PATH_RESULTS, 'mni_region_features.csv')
    if not os.path.exists(mni_path):
        raise FileNotFoundError(f"MNI features file not found at {mni_path}")
    return pd.read_csv(mni_path)

def load_desikan_killiany():
    lut_path = os.path.join(BASE_PATH_DATA, 'desikanKilliany.csv')
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT file not found at {lut_path}")
    return pd.read_csv(lut_path)

def load_atlas_coordinates(atlas_path, lut_df, nilearn_plotting):
    coords, labels = nilearn_plotting.find_parcellation_cut_coords(atlas_path, return_label_names=True)
    labels = pd.to_numeric(labels, errors='coerce')
    atlas_coord = pd.DataFrame({
        'roiNum': labels,
        'mni_x': coords[:, 0], 
        'mni_y': coords[:, 1], 
        'mni_z': coords[:, 2]
    })
    atlas_coord = pd.merge(lut_df, atlas_coord, on='roiNum', how='inner')
    return atlas_coord
