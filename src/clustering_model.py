import joblib
import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import warnings
from src import config

class ClusteringModel:
    def __init__(self) -> None:
        self.kmeans_model: Optional[KMeans] = None
        self.umap_model: Optional[umap.UMAP] = None
        self.best_params: Optional[Dict[str, Any]] = None

    def _optuna_objective(self, trial: optuna.Trial, data: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Objective function for Optuna to maximize silhouette score.
        Optimizes UMAP parameters AND number of clusters together (matching notebook).
        """
        warnings.filterwarnings('ignore')
        
        # Define hyperparameter search space (matching notebook)
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 
                                            config.UMAP_N_NEIGHBORS_RANGE[0], 
                                            config.UMAP_N_NEIGHBORS_RANGE[1]),
            'min_dist': trial.suggest_float('min_dist', 
                                           config.UMAP_MIN_DIST_RANGE[0], 
                                           config.UMAP_MIN_DIST_RANGE[1]),
            'n_components': trial.suggest_categorical('n_components', 
                                                     config.UMAP_N_COMPONENTS_OPTIONS),
            'metric': trial.suggest_categorical('metric', 
                                               config.UMAP_METRIC_OPTIONS),
            'n_clusters': trial.suggest_int('n_clusters', 
                                           config.OPTUNA_K_RANGE[0], 
                                           config.OPTUNA_K_RANGE[1]),
        }
        
        try:
            # UMAP embedding (matching notebook)
            reducer = umap.UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_components=params['n_components'],
                metric=params['metric'],
                random_state=config.RANDOM_STATE,
                verbose=False
            )
            embedding_result = reducer.fit_transform(data)
            embedding = np.asarray(embedding_result)

            # K-Means clustering on UMAP embedding (matching notebook)
            kmeans = KMeans(
                n_clusters=params['n_clusters'],
                random_state=config.RANDOM_STATE,
                n_init=10
            )
            labels = kmeans.fit_predict(embedding)

            # Return silhouette score on UMAP space (matching notebook)
            score = silhouette_score(embedding, labels)
            return float(score)

        except Exception:
            return -1.0

    def optimize_parameters(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Runs the Optuna study to find the best UMAP parameters and k together.
        Matches notebook's optimization approach.
        """
        print("Running Optuna optimization (UMAP + K-Means)...")
        print("Maximizing Silhouette Score with UMAP + K-Means")
        
        # Use TPESampler with fixed seed for reproducibility (matching notebook)
        sampler = TPESampler(seed=config.RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Optimize (matching notebook: 50 trials)
        study.optimize(
            lambda trial: self._optuna_objective(trial, data), 
            n_trials=config.OPTUNA_N_TRIALS,
            show_progress_bar=False
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        print(f"\nOPTIMIZATION COMPLETED")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"Trials: {len(study.trials)}")
        print(f"\nOPTIMAL PARAMETERS:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params

    def train_umap(self, data: Union[np.ndarray, pd.DataFrame], params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Trains UMAP model with optimized parameters and returns the embedding.
        Matches notebook's approach.
        """
        if params is None:
            params = self.best_params
        
        assert params is not None, "No parameters provided for UMAP training"
        
        print("Training UMAP model with optimized parameters...")
        self.umap_model = umap.UMAP(
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            n_components=params['n_components'],
            metric=params['metric'],
            random_state=config.RANDOM_STATE
        )
        
        # Fit and transform the data
        embedding_result = self.umap_model.fit_transform(data)
        embedding = np.asarray(embedding_result)
        
        # Save the UMAP model
        joblib.dump(self.umap_model, config.UMAP_MODEL_PATH)
        print("UMAP model trained and saved.")
        
        return embedding

    def train_kmeans(self, embedding: Union[np.ndarray, pd.DataFrame], n_clusters: int) -> None:
        """
        Trains the final KMeans model on the UMAP embedding.
        Matches notebook's approach.
        """
        print(f"Training final KMeans model with k={n_clusters} on UMAP embedding...")
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=config.RANDOM_STATE, 
            n_init=10
        )
        self.kmeans_model.fit(embedding)
        
        # Save the model
        joblib.dump(self.kmeans_model, config.KMEANS_MODEL_PATH)
        print("KMeans model trained and saved.")

    def predict(self, embedding: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts cluster for UMAP embedding using the saved KMeans model.
        Note: Input should be UMAP-transformed data, not raw features.
        """
        if self.kmeans_model is None:
            self.kmeans_model = joblib.load(config.KMEANS_MODEL_PATH)
        
        assert self.kmeans_model is not None, "KMeans model failed to load"
        return self.kmeans_model.predict(embedding)

    def transform_umap(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Applies the saved UMAP transformation to new data.
        This is needed before prediction.
        """
        if self.umap_model is None:
            self.umap_model = joblib.load(config.UMAP_MODEL_PATH)
        
        assert self.umap_model is not None, "UMAP model failed to load"
        result = self.umap_model.transform(data)
        return np.array(result)

