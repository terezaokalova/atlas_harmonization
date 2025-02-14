# mw_es_hup_config.py
import os

# Base directories
CODE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
BASE_PATH_RESULTS = os.path.join(CODE_DIRECTORY, '../results')
BASE_PATH_DATA = os.path.join(CODE_DIRECTORY, '../Data')

# Paths to data
DESIKAN_KILLIANY = os.path.join(BASE_PATH_DATA, 'aparc+aseg.nii.gz')
ATLAS_LUT = os.path.join(BASE_PATH_DATA, 'desikanKilliany.csv')

# Minimum number of patients per region to consider
MIN_PATIENTS = 5

# Feature columns of interest
FEATURE_COLUMNS = [
    'deltaRel_mean', 'thetaRel_mean', 'alphaRel_mean', 
    'betaRel_mean', 'gammaRel_mean', 'entropy_1min_mean', 'entropy_fullts_mean'
]

# Feature name mapping for visualizations
FEATURE_NAME_MAPPING = {
    'deltaRel_mean': 'delta',
    'thetaRel_mean': 'theta', 
    'alphaRel_mean': 'alpha',
    'betaRel_mean': 'beta',
    'gammaRel_mean': 'gamma',
    'entropy_1min_mean': 'entropy 1min',
    'entropy_fullts_mean': 'entropy full'
}
