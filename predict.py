import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Tuple, Dict, Any
from src.data_preprocessor import DataPreprocessor
from src.clustering_model import ClusteringModel
from src import config

def predict_new_sample(sample_data_dict: Dict[str, Any]) -> Tuple[int, np.ndarray, int]:
    """
    Loads all saved artifacts to process and predict a single new sample.
    Uses UMAP-first approach matching notebook.
    Auto-assigns patient ID and saves to all_patients_data.json.
    Returns cluster, 2D coordinates, and assigned patient_id.
    """
    print(f"--- PREDICTING NEW SAMPLE ---")
    
    # Add a dummy patient_id if not provided (preprocessor will drop it anyway)
    if 'patient_id' not in sample_data_dict:
        sample_data_dict['patient_id'] = 'TEMP_ID'
    
    # 1. Convert dict to DataFrame
    sample_df = pd.DataFrame([sample_data_dict])
    
    # 2. Load and apply preprocessing (frequency encode, impute, scale)
    # We use .transform() here, NOT .fit_transform()
    preprocessor = DataPreprocessor()
    sample_processed = preprocessor.transform(sample_df)
    
    # 3. Load models and predict
    model = ClusteringModel()
    
    # 3a. Transform through UMAP first (matching notebook approach)
    # This is CRITICAL: we must transform to UMAP space before clustering
    sample_embedding = model.transform_umap(sample_processed)
    
    # 3b. Predict cluster on UMAP embedding (not on raw features)
    cluster_labels = model.predict(sample_embedding)
    cluster = int(np.array(cluster_labels).flatten()[0])
    
    # 3c. Get 2D coordinates for visualization
    # Load the 2D UMAP model for visualization
    umap_2d_path = os.path.join(config.ARTIFACTS_DIR, "umap_2d_viz.joblib")
    umap_2d = joblib.load(umap_2d_path)
    sample_embedding_2d = umap_2d.transform(sample_processed)
    umap_coords_2d = np.asarray(sample_embedding_2d).flatten()
    
    # 4. Auto-assign patient ID and save to JSON
    json_path = os.path.join(config.ARTIFACTS_DIR, "all_patients_data.json")
    
    # Also save full patient details for newly predicted patients
    full_details_path = os.path.join(config.ARTIFACTS_DIR, "predicted_patients_details.json")
    
    # Load existing patients data
    try:
        with open(json_path, 'r') as f:
            all_patients = json.load(f)
    except FileNotFoundError:
        all_patients = []
    
    # Load existing predicted patient details
    try:
        with open(full_details_path, 'r') as f:
            predicted_details = json.load(f)
    except FileNotFoundError:
        predicted_details = {}
    
    # Get next patient ID (max existing ID + 1)
    if all_patients:
        next_patient_id = max(p['patient_id'] for p in all_patients) + 1
    else:
        next_patient_id = 0
    
    # Create new patient record for visualization
    new_patient_record = {
        'patient_id': next_patient_id,
        'x': float(umap_coords_2d[0]),
        'y': float(umap_coords_2d[1]),
        'cluster': int(cluster)
    }
    
    # Save full patient details (original input data)
    # Remove the temporary patient_id we added
    original_input = sample_data_dict.copy()
    if 'patient_id' in original_input and original_input['patient_id'] == 'TEMP_ID':
        del original_input['patient_id']
    
    predicted_details[str(next_patient_id)] = original_input
    
    # Append and save visualization data
    all_patients.append(new_patient_record)
    with open(json_path, 'w') as f:
        json.dump(all_patients, f, indent=2)
    
    # Save predicted patient details
    with open(full_details_path, 'w') as f:
        json.dump(predicted_details, f, indent=2)
    
    print(f"✅ New patient saved with ID: {next_patient_id}")
    print(f"   Total patients now: {len(all_patients)}")
    
    return cluster, umap_coords_2d, next_patient_id

if __name__ == "__main__":
    # This is the exact sample from your notebook's final cell
    new_patient_data = {
        'age': 55.0,
        'gender': 'Male',
        'ethnicity': 'Hispanic',
        'insurance_type': 'Private',
        'BMI': 28.0,
        'systolic_bp': 135.0,
        'diastolic_bp': 85.0,
        'heart_rate': 75.0,
        'cholesterol_total': 210.0,
        'blood_glucose': 110.0,
        'diabetes': 0.0,
        'hypertension': 1.0,
        'heart_disease': 0.0,
        'smoking_status': 'Never',
        'alcohol_consumption': 'Moderate',
        'doctor_visits_per_year': 3.0,
        'num_medications': 2.0,
        'medication_adherence': 0.9,
        'treatment_success_rate': 0.85,
        'patient_id': 'NEW_PATIENT_001' # This will be dropped
    }
    
    predicted_cluster, umap_coords, patient_id = predict_new_sample(new_patient_data)
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Assigned Patient ID: {patient_id}")
    print(f"Assigned Cluster: {predicted_cluster}")
    print(f"UMAP Coordinates (for plotting): {umap_coords}")
