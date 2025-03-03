#%% [Interactive Cell] Setup
import pickle
from pathlib import Path

#%% [Interactive Cell] Define functions 
def load_hup_data(data_dir):
    """
    Loads HUP data from a directory containing one folder per subject.
    Each subject folder should contain 20 pickle files, each representing a 1-minute epoch.
    
    Arguments:
        data_dir (str or Path): The path to the HUP data directory.
        
    Returns:
        dict: A dictionary with subject IDs as keys and values being a dictionary
              mapping the epoch file names to the loaded data.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    
    subject_data = {}
    # Iterate over each subject folder in the directory.
    for subject_folder in data_dir.iterdir():
        if subject_folder.is_dir():
            subject_id = subject_folder.name  # The folder name is the subject id.
            epoch_data = {}
            # Get all pkl files inside this subject folder.
            pkl_files = sorted(subject_folder.glob("*.pkl"))
            if len(pkl_files) != 20:
                print(f"Warning: Subject {subject_id} expected 20 pkl files, found {len(pkl_files)}")
            # Load each pickle file.
            for pkl_file in pkl_files:
                with open(pkl_file, "rb") as f:
                    epoch = pickle.load(f)
                # Use the file name as key; you could also extract an epoch number if the naming is consistent.
                epoch_data[pkl_file.name] = epoch
            subject_data[subject_id] = epoch_data
    return subject_data

def inspect_data_structure(data):
    """
    Inspects the structure of the loaded HUP data.
    
    Arguments:
        data (dict): Dictionary returned by `load_hup_data` containing loaded data.
    """
    print("Inspecting HUP data structure:")
    for subject_id, epochs in data.items():
        print(f"Subject: {subject_id} - Number of epochs loaded: {len(epochs)}")
        # For each epoch, print the file name and the type of the data loaded.
        for epoch_file, epoch_data in epochs.items():
            print(f"  Epoch file: {epoch_file} -> data type: {type(epoch_data)}")

# [Interactive Cell] Execute
DATA_PATH = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup"
# Load the data.
hup_data = load_hup_data(DATA_PATH)
# Inspect the structure.
inspect_data_structure(hup_data)

# %%
