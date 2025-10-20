import os

# --- Core Settings ---
RANDOM_STATE = 42

# --- File Paths ---
# Use os.path.join to make paths work on any operating system
DATA_FILE = os.path.join("data", "patients_with_missing.csv")
ARTIFACTS_DIR = "artifacts"

# Paths for saved model artifacts
FREQ_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "freq_encoder_maps.joblib")
IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "imputer.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
KMEANS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "kmeans_model.joblib")
UMAP_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "umap_model.joblib")
BEST_PARAMS_PATH = os.path.join(ARTIFACTS_DIR, "best_params.joblib")
COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "final_columns.joblib")

# --- Feature Engineering ---
# Columns from your notebook
HIGH_CARDINALITY_DROP_COLS = ['patient_id']
CATEGORICAL_COLS = ['gender', 'ethnicity', 'insurance_type', 'smoking_status', 'alcohol_consumption']

# --- Preprocessing ---
IMPUTER_N_NEIGHBORS = 5

# --- Model Training ---
OPTUNA_N_TRIALS = 50      # Number of trials to find best parameters
OPTUNA_K_RANGE = (2, 12)  # Range of k to test (matching notebook: 2-12 clusters)

# --- UMAP Parameters for Optimization ---
UMAP_N_NEIGHBORS_RANGE = (5, 50)           # Notebook: 5-50
UMAP_MIN_DIST_RANGE = (0.0, 0.2)           # Notebook: 0.0-0.2
UMAP_N_COMPONENTS_OPTIONS = [2, 3, 5, 8, 10]  # Notebook: 2, 3, 5, 8, 10
UMAP_METRIC_OPTIONS = ['euclidean', 'cosine', 'manhattan']  # Notebook metrics
