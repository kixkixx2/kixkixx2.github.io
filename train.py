from src.data_loader import load_data
from src.data_preprocessor import DataPreprocessor
from src.clustering_model import ClusteringModel
from src import config
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score

def main():
    print("--- STARTING TRAINING PIPELINE (NOTEBOOK APPROACH) ---")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load Data
    data = load_data(config.DATA_FILE)
    if data is None:
        print("Stopping pipeline: Data could not be loaded.")
        return

    # 2. Preprocess Data (frequency encode, impute, scale)
    # This will fit and save all preprocessors
    preprocessor = DataPreprocessor()
    data_processed = preprocessor.fit_transform(data)

    # 3. Modeling (UMAP-first approach matching notebook)
    model = ClusteringModel()
    
    # 3a. Optimize UMAP parameters + k together (matching notebook)
    best_params = model.optimize_parameters(data_processed)
    
    # Save best parameters for later use
    joblib.dump(best_params, config.BEST_PARAMS_PATH)
    print(f"Best parameters saved to {config.BEST_PARAMS_PATH}")
    
    # 3b. Train UMAP model with optimized parameters and get embedding
    embedding = model.train_umap(data_processed, best_params)
    
    # 3c. Train final KMeans model on UMAP embedding
    best_k = best_params['n_clusters']
    model.train_kmeans(embedding, best_k)
    
    # 4. Evaluate clustering quality (matching notebook)
    assert model.kmeans_model is not None, "KMeans model should be trained"
    labels = model.kmeans_model.labels_
    
    silhouette = silhouette_score(embedding, labels)
    db_index = davies_bouldin_score(embedding, labels)
    
    print("\n--- CLUSTERING QUALITY METRICS ---")
    print(f"Number of clusters: {best_k}")
    print(f"Silhouette Score: {silhouette:.4f}  (↑ better)")
    print(f"Davies-Bouldin Index: {db_index:.4f}  (↓ better)")
    
    # Cluster size breakdown
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    cluster_distribution = {}
    for cluster, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")
        cluster_distribution[int(cluster)] = {"count": int(count), "percentage": round(float(pct), 2)}
    
    # Save complete pipeline metadata for AI knowledge base
    import json
    pipeline_metadata = {
        "pipeline_description": "Patient clustering using UMAP dimensionality reduction and KMeans clustering",
        "total_patients": int(len(labels)),
        "num_features": int(data_processed.shape[1]),
        "preprocessing": {
            "steps": [
                {
                    "name": "Frequency Encoding",
                    "description": "Categorical variables encoded using frequency encoding (value replaced by occurrence frequency)",
                    "applied_to": ["gender", "ethnicity", "insurance_type", "smoking_status", "alcohol_consumption"]
                },
                {
                    "name": "Imputation",
                    "description": "Missing values handled with SimpleImputer using median strategy"
                },
                {
                    "name": "Scaling",
                    "description": "StandardScaler applied to normalize all features",
                    "formula": "z = (x - μ) / σ where μ is mean, σ is standard deviation"
                }
            ]
        },
        "dimensionality_reduction": {
            "algorithm": "UMAP",
            "full_name": "Uniform Manifold Approximation and Projection",
            "purpose": "Reduce high-dimensional feature space while preserving local and global structure",
            "parameters": {
                "n_neighbors": int(best_params['n_neighbors']),
                "min_dist": float(best_params['min_dist']),
                "n_components": int(best_params['n_components']),
                "metric": str(best_params['metric']),
                "random_state": int(config.RANDOM_STATE)
            },
            "theory": {
                "method": "Constructs high-dimensional graph representation using fuzzy simplicial set theory",
                "optimization": "Optimizes low-dimensional representation via cross-entropy minimization",
                "advantage": "Preserves topological structure better than linear methods like PCA"
            },
            "why_umap": [
                "Handles high-dimensional data better than PCA",
                "Faster than t-SNE for large datasets",
                "Better preservation of global structure",
                "Works well with cosine distance for medical features"
            ]
        },
        "clustering": {
            "algorithm": "KMeans",
            "n_clusters": int(best_k),
            "description": "KMeans clustering applied on UMAP-reduced data",
            "kmeans_theory": {
                "initialization": "Initialize k centroids randomly",
                "assignment": "Assign each point to nearest centroid using Euclidean distance",
                "update": "Recalculate centroids as mean of assigned points",
                "convergence": "Repeat until centroids don't change",
                "objective_formula": "minimize Σⱼ Σᵢ∈Cⱼ ||xᵢ - μⱼ||² where μⱼ is centroid of cluster j"
            }
        },
        "optimization": {
            "method": "Optuna (Bayesian Optimization with TPE Sampler)",
            "search_space": {
                "n_neighbors": list(config.UMAP_N_NEIGHBORS_RANGE),
                "min_dist": list(config.UMAP_MIN_DIST_RANGE),
                "n_components": config.UMAP_N_COMPONENTS_OPTIONS,
                "metric": config.UMAP_METRIC_OPTIONS,
                "n_clusters": list(config.OPTUNA_K_RANGE)
            },
            "optimization_metric": "Silhouette Score (maximize)",
            "n_trials": int(config.OPTUNA_N_TRIALS),
            "best_parameters": best_params
        },
        "evaluation_metrics": {
            "silhouette_score": {
                "value": round(float(silhouette), 4),
                "formula": "s(i) = (b(i) - a(i)) / max(a(i), b(i))",
                "description": "Measures how similar a point is to its own cluster vs other clusters",
                "range": "[-1, 1]",
                "interpretation": "Higher is better. Values > 0.7 are excellent, > 0.5 are good",
                "result_interpretation": "Excellent separation" if silhouette > 0.7 else "Good separation" if silhouette > 0.5 else "Moderate separation"
            },
            "davies_bouldin_index": {
                "value": round(float(db_index), 4),
                "formula": "DB = (1/k) Σ maxⱼ≠ᵢ((σᵢ + σⱼ) / d(cᵢ, cⱼ))",
                "description": "Ratio of within-cluster to between-cluster distances",
                "interpretation": "Lower is better. Values closer to 0 indicate better separation"
            }
        },
        "cluster_distribution": cluster_distribution,
        "formulas": {
            "standard_scaling": "z = (x - μ) / σ",
            "euclidean_distance": "d = √(Σ(xᵢ - yᵢ)²)",
            "cosine_similarity": "cos(θ) = (A·B) / (||A|| ||B||)",
            "silhouette_score": "s(i) = (b(i) - a(i)) / max(a(i), b(i))",
            "davies_bouldin_index": "DB = (1/k) Σ maxⱼ≠ᵢ((σᵢ + σⱼ) / d(cᵢ, cⱼ))",
            "kmeans_objective": "minimize Σⱼ Σᵢ∈Cⱼ ||xᵢ - μⱼ||²"
        },
        "visualization": {
            "method": "Separate 2D UMAP projection for scatter plots",
            "coordinates": "X and Y represent 2D embedding of 22-dimensional patient data",
            "coloring": "Color-coded by cluster assignment"
        },
        "prediction_pipeline": [
            "Apply same preprocessing (encoding, imputation, scaling)",
            "Transform through trained UMAP model",
            "Assign to nearest cluster centroid via KMeans",
            "Project to 2D for visualization"
        ]
    }
    
    # Save pipeline metadata
    metadata_path = os.path.join(config.ARTIFACTS_DIR, "pipeline_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(pipeline_metadata, f, indent=2)
    print(f"\nPipeline metadata saved to {metadata_path}")
    
    print("\n--- CREATING VISUALIZATION DATA ---")
    
    # Create a separate 2D UMAP for visualization purposes
    # (The clustering uses the full n_components dimensional embedding)
    import umap.umap_ as umap
    print("Creating 2D UMAP projection for visualization...")
    umap_2d = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=config.RANDOM_STATE
    )
    embedding_2d = umap_2d.fit_transform(data_processed)
    embedding_2d = np.asarray(embedding_2d)
    
    # Combine into a DataFrame with patient IDs if available
    # Use 1-based indexing to match CSV patient numbers (PAT_000001 -> ID 1)
    vis_df = pd.DataFrame({
        'patient_id': range(1, len(labels) + 1),  # Start from 1 instead of 0
        'x': embedding_2d[:, 0].tolist(),
        'y': embedding_2d[:, 1].tolist(),
        'cluster': labels.tolist()
    })
    
    # Save as a JSON file
    vis_data_path = os.path.join(config.ARTIFACTS_DIR, "all_patients_data.json")
    vis_df.to_json(vis_data_path, orient="records")
    
    # Also save the 2D UMAP model for prediction visualization
    umap_2d_path = os.path.join(config.ARTIFACTS_DIR, "umap_2d_viz.joblib")
    joblib.dump(umap_2d, umap_2d_path)
    
    print(f"Visualization data saved to {vis_data_path}")
    print(f"2D UMAP model saved to {umap_2d_path}")
    
    # 5. Generate cluster profiles for AI chatbot (Phase 1 of AI integration)
    print("\n--- GENERATING CLUSTER PROFILES FOR AI ---")
    cluster_profiles = generate_cluster_profiles(data, labels)
    profiles_path = os.path.join(config.ARTIFACTS_DIR, "cluster_profiles.json")
    
    import json
    with open(profiles_path, 'w') as f:
        json.dump(cluster_profiles, f, indent=2)
    
    print(f"Cluster profiles saved to {profiles_path}")
    print("--- TRAINING PIPELINE COMPLETE ---")
    print(f"All artifacts saved to '{config.ARTIFACTS_DIR}' directory.")

def generate_cluster_profiles(data: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Generate detailed statistical profiles for each cluster.
    This will be used by the AI chatbot to provide intelligent insights.
    """
    import json
    
    # Add cluster labels to the dataframe
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    profiles = {}
    
    for cluster_id in sorted(data_with_clusters['cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        n_patients = len(cluster_data)
        
        # Calculate statistics for this cluster
        profile = {
            "cluster_id": int(cluster_id),
            "name": f"Cluster {cluster_id}",
            "total_patients": int(n_patients),
            "percentage": round((n_patients / len(data_with_clusters)) * 100, 1),
            
            # Demographics
            "demographics": {
                "avg_age": round(cluster_data['age'].mean(), 1),
                "age_range": f"{int(cluster_data['age'].min())}-{int(cluster_data['age'].max())}",
                "gender_distribution": cluster_data['gender'].value_counts().to_dict(),
                "most_common_ethnicity": cluster_data['ethnicity'].mode().iloc[0] if len(cluster_data['ethnicity'].mode()) > 0 else "N/A",
                "most_common_insurance": cluster_data['insurance_type'].mode().iloc[0] if len(cluster_data['insurance_type'].mode()) > 0 else "N/A"
            },
            
            # Vital Signs
            "vital_signs": {
                "avg_bmi": round(cluster_data['BMI'].mean(), 1),
                "bmi_category": categorize_bmi(cluster_data['BMI'].mean()),
                "avg_systolic_bp": round(cluster_data['systolic_bp'].mean(), 1),
                "avg_diastolic_bp": round(cluster_data['diastolic_bp'].mean(), 1),
                "bp_status": categorize_blood_pressure(cluster_data['systolic_bp'].mean(), cluster_data['diastolic_bp'].mean()),
                "avg_heart_rate": round(cluster_data['heart_rate'].mean(), 1),
                "avg_cholesterol": round(cluster_data['cholesterol_total'].mean(), 1),
                "cholesterol_status": categorize_cholesterol(cluster_data['cholesterol_total'].mean()),
                "avg_glucose": round(cluster_data['blood_glucose'].mean(), 1),
                "glucose_status": categorize_glucose(cluster_data['blood_glucose'].mean())
            },
            
            # Health Conditions (prevalence %)
            "health_conditions": {
                "diabetes_rate": round((cluster_data['diabetes'].sum() / n_patients) * 100, 1),
                "hypertension_rate": round((cluster_data['hypertension'].sum() / n_patients) * 100, 1),
                "heart_disease_rate": round((cluster_data['heart_disease'].sum() / n_patients) * 100, 1),
                "smoking_distribution": cluster_data['smoking_status'].value_counts().to_dict(),
                "alcohol_distribution": cluster_data['alcohol_consumption'].value_counts().to_dict()
            },
            
            # Treatment & Care
            "treatment": {
                "avg_doctor_visits": round(cluster_data['doctor_visits_per_year'].mean(), 1),
                "avg_medications": round(cluster_data['num_medications'].mean(), 1),
                "avg_adherence_rate": round(cluster_data['medication_adherence'].mean() * 100, 1),
                "avg_treatment_success": round(cluster_data['treatment_success_rate'].mean() * 100, 1)
            },
            
            # Risk Assessment
            "risk_assessment": assess_cluster_risk(cluster_data),
            
            # Key Characteristics
            "key_characteristics": generate_key_characteristics(cluster_data, cluster_id)
        }
        
        profiles[f"cluster_{cluster_id}"] = profile
    
    return profiles

def categorize_bmi(bmi: float) -> str:
    """Categorize BMI into health categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def categorize_blood_pressure(systolic: float, diastolic: float) -> str:
    """Categorize blood pressure."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    else:
        return "Stage 2 Hypertension"

def categorize_cholesterol(cholesterol: float) -> str:
    """Categorize total cholesterol."""
    if cholesterol < 200:
        return "Desirable"
    elif cholesterol < 240:
        return "Borderline High"
    else:
        return "High"

def categorize_glucose(glucose: float) -> str:
    """Categorize fasting blood glucose."""
    if glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Prediabetes"
    else:
        return "Diabetes"

def assess_cluster_risk(cluster_data: pd.DataFrame) -> dict:
    """Assess overall risk level of the cluster."""
    # Calculate risk factors
    high_bp = (cluster_data['systolic_bp'] >= 140).sum() / len(cluster_data) * 100
    high_bmi = (cluster_data['BMI'] >= 30).sum() / len(cluster_data) * 100
    diabetes = (cluster_data['diabetes'] == 1).sum() / len(cluster_data) * 100
    heart_disease = (cluster_data['heart_disease'] == 1).sum() / len(cluster_data) * 100
    
    # Determine overall risk level
    risk_score = (high_bp + high_bmi + diabetes + heart_disease) / 4
    
    if risk_score < 25:
        risk_level = "Low"
        risk_description = "Generally healthy with minimal chronic conditions"
    elif risk_score < 50:
        risk_level = "Moderate"
        risk_description = "Some health concerns requiring monitoring"
    else:
        risk_level = "High"
        risk_description = "Multiple chronic conditions requiring active management"
    
    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "description": risk_description,
        "high_bp_percentage": round(high_bp, 1),
        "high_bmi_percentage": round(high_bmi, 1)
    }

def generate_key_characteristics(cluster_data: pd.DataFrame, cluster_id: int) -> list:
    """Generate human-readable key characteristics for the cluster."""
    characteristics = []
    
    # Age characteristic
    avg_age = cluster_data['age'].mean()
    if avg_age < 40:
        characteristics.append(f"Younger population (avg age: {int(avg_age)})")
    elif avg_age < 60:
        characteristics.append(f"Middle-aged population (avg age: {int(avg_age)})")
    else:
        characteristics.append(f"Older population (avg age: {int(avg_age)})")
    
    # BMI characteristic
    avg_bmi = cluster_data['BMI'].mean()
    if avg_bmi >= 30:
        characteristics.append(f"High obesity rate (avg BMI: {avg_bmi:.1f})")
    elif avg_bmi >= 25:
        characteristics.append(f"Overweight tendency (avg BMI: {avg_bmi:.1f})")
    else:
        characteristics.append(f"Healthy weight range (avg BMI: {avg_bmi:.1f})")
    
    # Chronic conditions
    diabetes_rate = (cluster_data['diabetes'].sum() / len(cluster_data)) * 100
    hypertension_rate = (cluster_data['hypertension'].sum() / len(cluster_data)) * 100
    heart_disease_rate = (cluster_data['heart_disease'].sum() / len(cluster_data)) * 100
    
    if diabetes_rate > 50:
        characteristics.append(f"High diabetes prevalence ({diabetes_rate:.0f}%)")
    if hypertension_rate > 50:
        characteristics.append(f"High hypertension prevalence ({hypertension_rate:.0f}%)")
    if heart_disease_rate > 30:
        characteristics.append(f"Significant heart disease ({heart_disease_rate:.0f}%)")
    
    # Treatment adherence
    avg_adherence = cluster_data['medication_adherence'].mean() * 100
    if avg_adherence >= 80:
        characteristics.append(f"Good medication adherence ({avg_adherence:.0f}%)")
    elif avg_adherence < 60:
        characteristics.append(f"Poor medication adherence ({avg_adherence:.0f}%)")
    
    return characteristics

if __name__ == "__main__":
    main()
