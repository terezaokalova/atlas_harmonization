# src/data_validator.py
import pandas as pd
import numpy as np
from typing import List, Tuple

class DataValidator:
    @staticmethod
    def validate_time_series(data: pd.DataFrame) -> bool:
        """Validate time series data"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Time series data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Time series data is empty")
        return True
    
    @staticmethod
    def validate_features(features: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate feature DataFrame"""
        missing_cols = set(expected_columns) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
    
    @staticmethod
    def compare_results(original: pd.DataFrame, new: pd.DataFrame, 
                       rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, str]:
        """Compare original and new results within tolerance"""
        try:
            pd.testing.assert_frame_equal(
                original, new,
                check_exact=False,
                rtol=rtol,
                atol=atol
            )
            return True, "Results match within tolerance"
        except AssertionError as e:
            return False, str(e)
