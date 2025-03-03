# %% [code]
import sys
import logging
import numpy as np
import pandas as pd

# Add the directory containing your modules to the Python path.
sys.path.append('/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/src/terez_scripts/clean_hup')

from clean_hup_data_loading import get_clean_hup_file_paths, load_epoch
from clean_hup_feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define configuration for feature extraction
config = {
    'preprocessing': {
        'sampling_frequency': 200
    },
    'features': {
        'spectral': {
            'bands': {
                'delta': [0.5, 4],
                'theta': [4, 8],
                'alpha': [8, 12],
                'beta': [12, 30],
                'gamma': [30, 80]
            }
        }
    }
}

# Set base path and choose a subject/epoch
base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
file_paths = get_clean_hup_file_paths(base_path)
subject = "sub-RID0031"
epoch_number = 1

# Get the file path for the desired subject and epoch
file_path = file_paths[subject][epoch_number]

# Load the epoch DataFrame using the file path
epoch_df = load_epoch(file_path)

# Print some info about the loaded DataFrame
print("Loaded epoch DataFrame:")
print(epoch_df.head())
print("DataFrame shape:", epoch_df.shape)

# Instantiate the FeatureExtractor and extract features
fe = FeatureExtractor(config)
features_df = fe.extract_features_from_epoch(epoch_df)

# Inspect the resulting features DataFrame
print("Extracted features DataFrame:")
print(features_df.head())
features_df.info()

# %%
