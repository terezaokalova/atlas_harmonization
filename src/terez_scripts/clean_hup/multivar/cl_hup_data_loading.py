import os
import pandas as pd
import numpy as np

def get_clean_hup_file_paths(base_path):
    """
    Scans the given base directory for subject folders and returns a dictionary
    mapping subject identifiers to a dictionary of epoch file paths.
    
    Only files starting with "interictal_eeg_bipolar_clean_" are included.
    If a subject has a different number than 20, a warning is printed.
    
    Returns:
        dict: {
            'sub-RID0031': {1: '/path/to/epoch1.pkl', 2: '/path/to/epoch2.pkl', ...},
            'sub-RID0032': { ... },
            ...
        }
    """
    subjects_dict = {}
    subject_dirs = sorted([d for d in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, d)) and d.startswith("sub-")])
    
    for subject in subject_dirs:
        subject_path = os.path.join(base_path, subject)
        # Only include raw epoch files that start with the expected prefix
        pkl_files = sorted([f for f in os.listdir(subject_path)
                            if f.startswith("interictal_eeg_bipolar_clean_") and f.endswith('.pkl')])
        if len(pkl_files) != 20:
            print(f"WARNING: {subject} has {len(pkl_files)} files instead of 20.")
        subjects_dict[subject] = {}
        for idx, filename in enumerate(pkl_files, start=1):
            file_path = os.path.join(subject_path, filename)
            subjects_dict[subject][idx] = file_path
    return subjects_dict

def load_epoch(file_path):
    """
    Loads a single epoch pickle file and returns a DataFrame.
    Assumes the pickle file is a dictionary with keys:
      - 'metadata': a DataFrame (one row per electrode)
      - 'data': a DataFrame with shape (time, electrodes) or a list/array of 1D EEG signals.
    This function ensures that each row in the resulting DataFrame has a "data" entry 
    that is a 1D numpy array of the full time series.
    """
    obj = pd.read_pickle(file_path)
    if isinstance(obj, dict):
        df = obj.get('metadata').copy()
        data_obj = obj.get('data')
        if isinstance(data_obj, pd.DataFrame):
            # If there are more rows than columns, assume rows are time stamps
            if data_obj.shape[0] > data_obj.shape[1]:
                data_obj = data_obj.T
            # Now each row corresponds to one electrode's full time series.
            df['data'] = data_obj.apply(lambda row: row.values, axis=1)
        else:
            # If it's not a DataFrame, convert each element to a numpy array.
            df['data'] = [np.asarray(x) for x in data_obj]
        return df
    return obj
# def load_epoch(file_path):
#     """
#     Loads a single epoch pickle file and returns the corresponding DataFrame.
    
#     The returned DataFrame is a metadata table for that epoch,
#     with each row corresponding to one electrode (the first column contains electrode labels)
#     and a column 'data' containing the EEG signal.
#     """
#     return pd.read_pickle(file_path)

if __name__ == '__main__':
    base_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean"
    file_paths = get_clean_hup_file_paths(base_path)
    print("Found file paths for subjects:")
    for subject, epochs in file_paths.items():
        print(f"{subject}: {len(epochs)} epochs")
