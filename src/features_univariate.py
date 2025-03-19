#%%
import os
import pandas as pd
import numpy as np
import h5py
from process_ieeg import IEEGClipProcessor
from IPython import embed
from pathlib import Path
from utils import in_parallel, catch22_single_series, compute_psd_all_channels_parallel, fooof_single_series
from fooof import FOOOF
import time

#%%
class UnivariateFeatures(IEEGClipProcessor):
    def __init__(self, subject_id: str):
        super().__init__()
        self.subject_id = subject_id
        self.ieeg_processed  = next(self.project_root.joinpath('data', 'derivatives').rglob(f'{subject_id}/**/interictal_ieeg_processed.h5'))
        self.ieeg_processed_bipolar, self.coordinates, self.sampling_frequency = self.load_ieeg()
        self.psd_df = compute_psd_all_channels_parallel(self.ieeg_processed_bipolar, self.sampling_frequency)

    def load_ieeg(self):
        if not self.ieeg_processed.exists():
            raise FileNotFoundError(f"No iEEG processed file found for subject {self.subject_id}")
        
        with h5py.File(self.ieeg_processed, 'r') as f:
                coordinates_data = f['/bipolar_montage/coordinates']
                ieeg_data = f['/bipolar_montage/ieeg']
                ieeg = pd.DataFrame(ieeg_data, columns=ieeg_data.attrs['channels_labels'])
                sampling_frequency = ieeg_data.attrs['sampling_rate']
                coordinates = pd.concat([
                    pd.DataFrame(coordinates_data, columns=['x', 'y', 'z'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['original_labels'], columns=['orig_labels'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roi'], columns=['roi'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roiNum'], columns=['roiNum'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['spared'], columns=['spared'], index=coordinates_data.attrs['labels'])
                ], axis=1)
        
        return ieeg, coordinates, sampling_frequency
    
    # add univariate features here
    
    def catch22_features(self):
        channels = self.ieeg_processed_bipolar.columns
        time_series_list = [self.ieeg_processed_bipolar[channel].values for channel in channels]

        results = in_parallel(catch22_single_series, time_series_list, verbose=True)

        feature_names = results[0]['names']
        feature_values = np.array([result['values'] for result in results])
        results_df = pd.DataFrame(feature_values, index=channels, columns=feature_names)

        return results_df

    # def catch22_features_test(self, max_channels=5):
    #     """Test version with sequential processing and clear progress"""
    #     channels = self.ieeg_processed_bipolar.columns[:max_channels]
    #     results = []
        
    #     print(f"Testing with {len(channels)} channels sequentially:")
    #     for i, channel in enumerate(channels):
    #         print(f"Processing channel {i+1}/{len(channels)}: {channel}")
    #         series = self.ieeg_processed_bipolar[channel].values
    #         start = time.time()
    #         result = catch22_single_series(series)
    #         elapsed = time.time() - start
    #         print(f"  Completed in {elapsed:.2f} seconds")
    #         results.append(result)
        
    #     return results

    def fooof_features(self):
        channels = self.ieeg_processed_bipolar.columns
        freqs = self.psd_df.index.values

        inputs = [(freqs, self.psd_df[channel].values) for channel in channels]

        results = in_parallel(fooof_single_series, inputs, verbose=True)

        return pd.DataFrame(results, index=channels)

    def bandpower_features(self):
        bands = {
            'delta': [1, 4],
            'theta': [4, 8], 
            'alpha': [8, 13],
            'beta': [13, 30],
            'gamma': [30, 100]
        }
        
        results = {}
        freqs = self.psd_df.index.values
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            results[band_name] = self.psd_df.loc[mask].mean()
            
        return pd.DataFrame(results)

    #def entropy_features(self):
    #    pass


#%%
if __name__ == "__main__":
    subject_id = "sub-RID0031"
    features = UnivariateFeatures(subject_id)
    #print(features.catch22_features())
    #print(features.ieeg_processed_bipolar.columns)
    #print(features.catch22_features(features.ieeg_processed_bipolar.columns[0]))
    #print(catch22_single_series(features.ieeg_processed_bipolar[features.ieeg_processed_bipolar.columns[0]].values))
    print(features.fooof_features().head())

