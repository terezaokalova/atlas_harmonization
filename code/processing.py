import pandas as pd
import os

class DataProcessor:
    def __init__(self, dataset_name):
        """Initialize DataProcessor with dataset name (HUP or MNI)"""
        self.dataset_name = dataset_name.upper()  
        self.data = self.load_dataset()
        
        # Initialize other attributes after data is loaded
        if self.data:
            self.initialize_attributes()
        else:
            raise ValueError(f"Failed to load {dataset_name} dataset")

    def load_dataset(self):
        """Load dataset files based on dataset name"""
        prefix = self.dataset_name.lower()
        base_path = self.dataset_name  
        
        # Define files we need to load
        files = {
            'elec': f'{prefix}_catch22_feats_elec.csv',
            'region': f'{prefix}_catch22_feats_region.csv',
            'region_avg': f'{prefix}_catch22_feats_region_avg.csv',
            'univar': f'{prefix}_univar_feats_raw.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = os.path.join(base_path, filename)
            try:
                data[key] = pd.read_csv(filepath)
                print(f"Successfully loaded {filepath}")
            except FileNotFoundError:
                print(f"Error: File not found - {filepath}")
                return None
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                return None
                
        return data

    def initialize_attributes(self):
        """Initialize class attributes after data is loaded"""
        self.data_attributes = {
            'elec': {
                'features': [col for col in self.data['elec'].columns if col != 'electrode'],
                'identifiers': ['electrode']
            },
            'region': {
                'features': [col for col in self.data['region'].columns if col not in ['patient_id', 'roiNum']],
                'identifiers': ['patient_id', 'roiNum', 'roi']
            },
            'region_avg': {
                'features': [col for col in self.data['region_avg'].columns if col not in ['roiNum', 'roi']],
                'identifiers': ['roiNum', 'roi']
            },
            'univar': {
                'features': list(self.data['univar'].columns),
                'identifiers': []  
            }
        }

        if 'Unnamed: 0' in self.data['univar'].columns:
            self.data['univar'].rename(columns={'Unnamed: 0': 'electrode'}, inplace=True)

        # Store counts 
        self.feature_counts = {key: len(attr['features']) for key, attr in self.data_attributes.items()}

    def get_df(self, df_type):
        """Get a specific dataframe by type.
        
        Args:
            df_type (str): One of 'elec', 'region', 'region_avg', 'univar_raw'
        """
        return self.data.get(df_type)

    def get_identifiers(self, df_type):
        """Get identifiers for a specific dataframe type."""
        return self.data_attributes[df_type]['identifiers'] if df_type in self.data_attributes else []

    def compare_columns(self):
        """Compare columns across all available dataframes."""
        # Convert columns to sets, skipping None dataframes
        col_sets = {
            name: set(df.columns) 
            for name, df in self.data.items() 
            if df is not None
        }
        
        # Print unique columns for each dataframe
        for name, cols in col_sets.items():
            other_cols = set().union(*(
                other_cols for other_name, other_cols in col_sets.items() 
                if other_name != name
            ))
            unique_cols = cols - other_cols
            print(f"\nColumns unique to {name}:", unique_cols)
        
        common_cols = set.intersection(*col_sets.values())
        print("\nColumns common to all DataFrames:", common_cols)



if __name__ == "__main__":
   
   mni = DataProcessor('mni')
   hup = DataProcessor('hup')

   print(hup.get_df('region').shape, mni.get_df('region').shape)


