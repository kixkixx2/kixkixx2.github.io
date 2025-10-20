import pandas as pd
from typing import Optional
from src import config

def load_data(file_path: str = config.DATA_FILE) -> Optional[pd.DataFrame]:
    """Loads the CSV data from the specified path."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
