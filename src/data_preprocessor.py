import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src import config

class DataPreprocessor:
    def __init__(self) -> None:
        self.freq_encoding_maps: Dict[str, Dict[Any, float]] = {}
        self.imputer = KNNImputer(n_neighbors=config.IMPUTER_N_NEIGHBORS)
        self.scaler = StandardScaler()
        self.numerical_cols: List[str] = []
        self.final_columns: List[str] = []

    def _frequency_encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Applies frequency encoding based on the training set."""
        df_processed = df.copy()
        for col in config.CATEGORICAL_COLS:
            if fit:
                # 1. Create and save the frequency map
                value_counts = df_processed[col].value_counts(dropna=False)
                freq_map = (value_counts / len(df_processed)).to_dict()
                self.freq_encoding_maps[col] = freq_map
            
            # 2. Apply the saved map
            # .get(col, {}) handles maps that might be missing (e.g., during prediction)
            # NaN values will be handled by KNNImputer later
            df_processed[col] = df_processed[col].map(self.freq_encoding_maps.get(col, {}))
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits all preprocessors and transforms the training data."""
        print("Fitting preprocessor...")
        # 1. Drop high cardinality columns
        df_processed = df.drop(columns=config.HIGH_CARDINALITY_DROP_COLS)
        
        # 2. Frequency encode categorical columns
        df_processed = self._frequency_encode(df_processed, fit=True)
        
        # 3. Identify numerical columns
        self.numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # 4. Fit and transform imputer
        df_processed[self.numerical_cols] = self.imputer.fit_transform(df_processed[self.numerical_cols])
        
        # 5. Fit and transform scaler
        df_processed[self.numerical_cols] = self.scaler.fit_transform(df_processed[self.numerical_cols])
        
        # 6. Save the final column order
        self.final_columns = df_processed.columns.tolist()
        
        # 7. Save all fitted artifacts
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(self.freq_encoding_maps, config.FREQ_ENCODER_PATH)
        joblib.dump(self.imputer, config.IMPUTER_PATH)
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.final_columns, config.COLUMNS_PATH)
        
        print("Preprocessor fitted and artifacts saved.")
        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using saved (fitted) preprocessors."""
        print("Transforming new data...")
        # 1. Load all artifacts
        self.freq_encoding_maps = joblib.load(config.FREQ_ENCODER_PATH)
        self.imputer = joblib.load(config.IMPUTER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.final_columns = joblib.load(config.COLUMNS_PATH)
        
        # 2. Drop high cardinality columns
        df_processed = df.drop(columns=config.HIGH_CARDINALITY_DROP_COLS)
        
        # 3. Apply saved frequency encoding
        df_processed = self._frequency_encode(df_processed, fit=False)
        
        # 4. Identify numerical columns
        self.numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        # 5. Apply saved imputer
        df_processed[self.numerical_cols] = self.imputer.transform(df_processed[self.numerical_cols])
        
        # 6. Apply saved scaler
        df_processed[self.numerical_cols] = self.scaler.transform(df_processed[self.numerical_cols])
        
        # 7. Ensure column order matches training data
        df_processed = df_processed[self.final_columns]
        
        print("New data transformed.")
        return df_processed
