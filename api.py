from flask import Flask, request, jsonify, send_from_directory
import os
import json
from typing import Optional
try:
    from flask_cors import CORS  # type: ignore
    cors_available = True
except ImportError:
    cors_available = False
from predict import predict_new_sample
from src import config
from dotenv import load_dotenv
import google.generativeai as genai  # type: ignore

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'AIzaSyDummyKeyReplaceMeWithRealKey123456789':
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        # Use the correct model name from available models (as of Oct 2025)
        # Stable models: models/gemini-2.5-flash, models/gemini-flash-latest
        # Fast & cost-effective: models/gemini-2.5-flash-lite
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')  # type: ignore
        print("✅ Gemini AI configured successfully with models/gemini-2.5-flash")
    except Exception as e:
        gemini_model = None
        print(f"⚠️  WARNING: Failed to configure Gemini AI: {str(e)}")
        print("   To see available models, run: python check_gemini_models.py")
else:
    gemini_model = None
    print("⚠️  WARNING: GEMINI_API_KEY not found or is dummy. Chat feature will be disabled.")
    print("   Get your FREE API key from: https://aistudio.google.com/app/apikey")

# Load cluster profiles (the "brain" for AI)
CLUSTER_PROFILES = {}
try:
    profiles_path = os.path.join(config.ARTIFACTS_DIR, "cluster_profiles.json")
    with open(profiles_path, 'r') as f:
        CLUSTER_PROFILES = json.load(f)
    print(f"✅ Loaded cluster profiles: {len(CLUSTER_PROFILES)} clusters")
except FileNotFoundError:
    print("⚠️  WARNING: cluster_profiles.json not found. Run train.py first.")

# Load full patient data for AI context
FULL_PATIENT_DATA = None
ALL_PATIENTS_VIZ = None

def load_patient_data():
    """Load complete patient data including CSV and JSON files"""
    global FULL_PATIENT_DATA, ALL_PATIENTS_VIZ
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load CSV with full patient details
        csv_path = config.DATA_FILE
        df = pd.read_csv(csv_path)
        
        # Load visualization data (includes cluster assignments and coordinates)
        viz_path = os.path.join(config.ARTIFACTS_DIR, "all_patients_data.json")
        with open(viz_path, 'r') as f:
            ALL_PATIENTS_VIZ = json.load(f)
        
        # Merge data: Add cluster info to CSV data
        # Create a mapping of patient_id to cluster
        cluster_map = {p['patient_id']: p['cluster'] for p in ALL_PATIENTS_VIZ}
        
        # Convert patient_id in CSV to numeric (remove PAT_ prefix)
        df['patient_num_id'] = df['patient_id'].str.replace('PAT_', '').str.lstrip('0').astype(int)
        df['cluster'] = df['patient_num_id'].map(cluster_map)
        
        # Convert to dict for AI access
        FULL_PATIENT_DATA = df.to_dict('records')
        
        print(f"✅ Loaded full patient data: {len(FULL_PATIENT_DATA)} patients with {len(df.columns)} features")
        return True
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load patient data: {str(e)}")
        return False

# Load data on startup
load_patient_data()

app = Flask(__name__, static_folder='frontend')
if cors_available:
    CORS(app)  # type: ignore  # Enable CORS for frontend requests

# Serve frontend
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return send_from_directory('frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict cluster for a new patient.
    Expects a JSON payload with patient data.
    Auto-assigns patient ID, saves to JSON, and returns results.
    """
    try:
        # 1. Get the patient data from the request
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'error': 'No data provided. Please send patient data as JSON.'
            }), 400
        
        # 2. Pass the data to the prediction function (now returns patient_id too)
        cluster, umap_coords, patient_id = predict_new_sample(patient_data)
        
        # 3. Return the results as JSON
        response = {
            'success': True,
            'patient_id': int(patient_id),
            'cluster': int(cluster),
            'umap_coordinates': {
                'x': float(umap_coords[0]),
                'y': float(umap_coords[1])
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Handle any errors gracefully
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Clustering API is running'
    }), 200

@app.route('/api/get_all_patients', methods=['GET'])
def get_all_patients():
    """
    API endpoint to send all training patient data 
    (UMAP coords + cluster) for frontend visualization.
    """
    try:
        # Define the path to the JSON file
        file_name = "all_patients_data.json"
        
        # 'send_from_directory' is the secure way to send a static file
        return send_from_directory(
            config.ARTIFACTS_DIR, 
            file_name, 
            as_attachment=False
        )
        
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error': 'Visualization data file not found. Run train.py first.'
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_patient/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """
    API endpoint to get full details for a specific patient.
    Handles both original patients (from CSV) and newly predicted patients (from JSON).
    """
    try:
        import pandas as pd
        import json
        import math
        
        # Load cluster assignment from visualization JSON
        vis_path = os.path.join(config.ARTIFACTS_DIR, "all_patients_data.json")
        with open(vis_path, 'r') as f:
            vis_data = json.load(f)
        
        # Find patient in visualization data
        patient_vis = next((p for p in vis_data if p['patient_id'] == patient_id), None)
        
        if not patient_vis:
            return jsonify({
                'success': False,
                'error': f'Patient ID {patient_id} not found in dataset'
            }), 404
        
        # Try to load from CSV (original patients)
        df = pd.read_csv(config.DATA_FILE)
        
        # Use 1-based indexing: patient_id 1 = row 0, patient_id 2 = row 1, etc.
        if 1 <= patient_id <= len(df):
            # Original patient from CSV (subtract 1 to convert to 0-based index)
            patient_data = df.iloc[patient_id - 1].to_dict()
            patient_data['patient_id'] = patient_id
            
            # Replace NaN values with None (becomes null in JSON)
            for key, value in patient_data.items():
                if isinstance(value, float) and math.isnan(value):
                    patient_data[key] = None
        else:
            # Newly predicted patient - load from predicted_patients_details.json
            details_path = os.path.join(config.ARTIFACTS_DIR, "predicted_patients_details.json")
            try:
                with open(details_path, 'r') as f:
                    predicted_details = json.load(f)
                
                # Get patient details from saved predictions
                if str(patient_id) in predicted_details:
                    patient_data = predicted_details[str(patient_id)]
                    patient_data['patient_id'] = patient_id
                    patient_data['note'] = 'This patient was added through prediction.'
                else:
                    # Fallback if details not found
                    patient_data = {
                        'patient_id': patient_id,
                        'note': 'This is a newly predicted patient. Input details not available.'
                    }
            except FileNotFoundError:
                # Fallback if file doesn't exist
                patient_data = {
                    'patient_id': patient_id,
                    'note': 'This is a newly predicted patient. Input details not available.'
                }
        
        # Add cluster and coordinates
        patient_data['cluster'] = patient_vis['cluster']
        patient_data['x'] = patient_vis['x']
        patient_data['y'] = patient_vis['y']
        
        return jsonify({
            'success': True,
            'patient': patient_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    AI chatbot endpoint that provides intelligent insights about patient clusters.
    Expects: { "message": "user question", "cluster_id": 0 or 1 (optional) }
    Returns: { "success": true, "response": "AI response" }
    """
    try:
        if not gemini_model:
            return jsonify({
                'success': False,
                'error': 'AI chatbot is not configured. Please add GEMINI_API_KEY to .env file.'
            }), 503
        
        data = request.get_json()
        user_message = data.get('message', '')
        cluster_id = data.get('cluster_id', None)
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Build the AI prompt with domain knowledge
        prompt = build_ai_prompt(user_message, cluster_id)
        
        # Get response from Gemini AI with generation config to limit length
        try:
            # Configure generation to encourage shorter responses
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 300,  # Limit response length (approx 60-80 words)
            }
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config  # type: ignore
            )
            ai_response = response.text
            
            # Optional: Warn if response is too long (for monitoring)
            word_count = len(ai_response.split())
            if word_count > 100:
                print(f"⚠️ AI response is long ({word_count} words). Consider adjusting prompt.")
            
        except Exception as gemini_error:
            # Handle Gemini API errors (quota, rate limits, etc.)
            error_message = str(gemini_error)
            
            # Check for quota errors
            if '429' in error_message or 'quota' in error_message.lower():
                return jsonify({
                    'success': False,
                    'error': 'API quota exceeded. Please try again later or check your API limits at https://ai.google.dev/gemini-api/docs/rate-limits'
                }), 429
            
            # Other API errors
            return jsonify({
                'success': False,
                'error': f'AI service error: {error_message[:200]}'
            }), 500
        
        return jsonify({
            'success': True,
            'response': ai_response
        }), 200
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'AI service error: {str(e)}'
        }), 500

def get_dataset_statistics():
    """Generate comprehensive dataset statistics for AI context"""
    if not FULL_PATIENT_DATA:
        return {
            "total_patients": 0,
            "error": "Patient data not loaded"
        }
    
    try:
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(FULL_PATIENT_DATA)
        
        # Helper function to safely get numeric stats
        def safe_stats(column_name):
            if column_name in df.columns:
                col = df[column_name].replace([np.inf, -np.inf], np.nan)
                return {
                    "mean": float(col.mean()) if not col.isna().all() else 0,
                    "min": float(col.min()) if not col.isna().all() else 0,
                    "max": float(col.max()) if not col.isna().all() else 0,
                    "median": float(col.median()) if not col.isna().all() else 0
                }
            return {"mean": 0, "min": 0, "max": 0, "median": 0}
        
        # Helper function to safely get percentage
        def safe_percentage(column_name):
            if column_name in df.columns:
                col = df[column_name].replace([np.inf, -np.inf], np.nan)
                return float(col.mean() * 100) if not col.isna().all() else 0
            return 0
        
        stats = {
            "total_patients": len(df),
            "cluster_distribution": df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {},
            "age": safe_stats('age'),
            "gender_distribution": df['gender'].value_counts().to_dict() if 'gender' in df.columns else {},
            "diabetes_rate": safe_percentage('diabetes'),
            "hypertension_rate": safe_percentage('hypertension'),
            "heart_disease_rate": safe_percentage('heart_disease'),
            "average_bmi": safe_stats('BMI')['mean'],
            "average_medications": safe_stats('num_medications')['mean']
        }
        
        return stats
        
    except Exception as e:
        print(f"Error in get_dataset_statistics: {str(e)}")
        return {
            "total_patients": 0,
            "error": f"Failed to compute statistics: {str(e)}"
        }

def search_patients_by_criteria(criteria: dict):
    """Search patients matching specific criteria"""
    if not FULL_PATIENT_DATA:
        return []
    
    try:
        import pandas as pd
        df = pd.DataFrame(FULL_PATIENT_DATA)
        
        # Apply filters based on criteria
        filtered = df.copy()
        
        if 'cluster' in criteria and 'cluster' in df.columns:
            filtered = filtered[filtered['cluster'] == criteria['cluster']]
        
        if 'min_age' in criteria and 'age' in df.columns:
            filtered = filtered[filtered['age'] >= criteria['min_age']]
        
        if 'max_age' in criteria and 'age' in df.columns:
            filtered = filtered[filtered['age'] <= criteria['max_age']]
        
        if 'diabetes' in criteria and 'diabetes' in df.columns:
            filtered = filtered[filtered['diabetes'] == criteria['diabetes']]
        
        if 'hypertension' in criteria and 'hypertension' in df.columns:
            filtered = filtered[filtered['hypertension'] == criteria['hypertension']]
        
        return filtered.head(5).to_dict('records')  # Return top 5 matches
        
    except Exception as e:
        print(f"Error in search_patients_by_criteria: {str(e)}")
        return []

def get_patient_by_id(patient_id: int) -> Optional[dict]:
    """
    Get a specific patient's data by their ID.
    Returns patient data including cluster assignment, or None if not found.
    """
    if not FULL_PATIENT_DATA:
        return None
    
    try:
        import pandas as pd
        df = pd.DataFrame(FULL_PATIENT_DATA)
        
        # Patient IDs are 1-based, so subtract 1 for 0-based index
        if 1 <= patient_id <= len(df):
            patient = df.iloc[patient_id - 1].to_dict()
            
            # Add the patient_id to the result
            patient['patient_id'] = patient_id
            
            # Convert any NaN values to None
            import math
            for key, value in patient.items():
                if isinstance(value, float) and math.isnan(value):
                    patient[key] = None
            
            return patient
        else:
            return None
            
    except Exception as e:
        print(f"Error in get_patient_by_id: {str(e)}")
        return None

def load_pipeline_metadata() -> dict:
    """
    Load complete pipeline metadata from saved artifacts.
    This includes all algorithms, parameters, formulas, and actual results.
    """
    try:
        metadata_path = os.path.join(config.ARTIFACTS_DIR, "pipeline_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            # Fallback minimal metadata if file doesn't exist
            print(f"⚠️  WARNING: {metadata_path} not found. Using minimal metadata.")
            return {
                "pipeline_description": "Patient clustering system (metadata not yet generated - run train.py)",
                "total_patients": 0,
                "num_features": 22,
                "error": "Pipeline metadata not found. Please run train.py to generate complete pipeline information."
            }
    
    except Exception as e:
        print(f"Error loading pipeline metadata: {str(e)}")
        return {
            "pipeline_description": "Patient clustering system",
            "error": f"Failed to load metadata: {str(e)}"
        }

def load_system_codebase() -> str:
    """
    Load and document the complete system codebase for AI understanding.
    This gives the AI full knowledge of implementation details.
    """
    try:
        codebase_summary = "\n\n=== COMPLETE SYSTEM CODEBASE ===\n"
        codebase_summary += "The AI has access to the ENTIRE codebase implementation:\n\n"
        
        # Define the files to read (main system files)
        files_to_read = [
            ("src/config.py", "Configuration settings and paths"),
            ("src/data_loader.py", "Data loading logic"),
            ("src/data_preprocessor.py", "Preprocessing pipeline (encoding, imputation, scaling)"),
            ("src/clustering_model.py", "UMAP + KMeans implementation and optimization"),
            ("train.py", "Main training pipeline"),
            ("predict.py", "Prediction pipeline for new patients"),
            ("api.py", "Flask API and chatbot logic"),
        ]
        
        for filename, description in files_to_read:
            file_path = os.path.join(os.path.dirname(__file__), filename) if filename != "api.py" else __file__
            
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Count lines and functions
                    lines = code_content.split('\n')
                    num_lines = len(lines)
                    
                    # Extract key functions/classes (simple detection)
                    functions = [line.strip() for line in lines if line.strip().startswith('def ') or line.strip().startswith('class ')]
                    
                    codebase_summary += f"\n📄 **{filename}** ({description})\n"
                    codebase_summary += f"   Lines: {num_lines} | Functions/Classes: {len(functions)}\n"
                    
                    if functions:
                        codebase_summary += f"   Key Components:\n"
                        for func in functions[:5]:  # Show first 5
                            func_name = func.replace('def ', '').replace('class ', '').split('(')[0].split(':')[0]
                            codebase_summary += f"     - {func_name}\n"
                        if len(functions) > 5:
                            codebase_summary += f"     ... and {len(functions) - 5} more\n"
                    
                    # Add key logic snippets (for important files)
                    if filename == "src/clustering_model.py":
                        codebase_summary += "\n   🔍 KEY IMPLEMENTATION DETAILS:\n"
                        codebase_summary += "   - Uses Optuna for hyperparameter optimization\n"
                        codebase_summary += "   - Optimizes UMAP parameters + k-means clusters together\n"
                        codebase_summary += "   - Evaluation: Silhouette score (maximize)\n"
                        codebase_summary += "   - UMAP reducer trained first, then KMeans on embedding\n"
                    
                    elif filename == "src/data_preprocessor.py":
                        codebase_summary += "\n   🔍 KEY IMPLEMENTATION DETAILS:\n"
                        codebase_summary += "   - Frequency encoding for categorical variables\n"
                        codebase_summary += "   - SimpleImputer with median strategy\n"
                        codebase_summary += "   - StandardScaler for normalization\n"
                        codebase_summary += "   - All transformers saved as artifacts\n"
                    
                    elif filename == "train.py":
                        codebase_summary += "\n   🔍 KEY WORKFLOW:\n"
                        codebase_summary += "   1. Load data from CSV\n"
                        codebase_summary += "   2. Preprocess (encode, impute, scale)\n"
                        codebase_summary += "   3. Optimize UMAP + KMeans parameters\n"
                        codebase_summary += "   4. Train final models\n"
                        codebase_summary += "   5. Evaluate with metrics\n"
                        codebase_summary += "   6. Save artifacts + generate profiles\n"
                        codebase_summary += "   7. Create 2D visualization\n"
                    
                    elif filename == "predict.py":
                        codebase_summary += "\n   🔍 KEY WORKFLOW:\n"
                        codebase_summary += "   1. Load saved preprocessors and models\n"
                        codebase_summary += "   2. Apply same preprocessing to new data\n"
                        codebase_summary += "   3. Transform through UMAP\n"
                        codebase_summary += "   4. Predict cluster with KMeans\n"
                        codebase_summary += "   5. Project to 2D for visualization\n"
                        codebase_summary += "   6. Assign patient ID and save\n"
                
            except Exception as file_error:
                codebase_summary += f"\n   ⚠️ Error reading {filename}: {str(file_error)}\n"
        
        codebase_summary += "\n\n💡 **AI's Understanding:**\n"
        codebase_summary += "The AI now knows:\n"
        codebase_summary += "✓ Every function and class in the system\n"
        codebase_summary += "✓ The complete workflow from data loading to prediction\n"
        codebase_summary += "✓ Implementation details of UMAP and KMeans\n"
        codebase_summary += "✓ How preprocessing works (encoding, imputation, scaling)\n"
        codebase_summary += "✓ How optimization is performed (Optuna TPE sampler)\n"
        codebase_summary += "✓ How predictions are made for new patients\n"
        codebase_summary += "✓ How the Flask API serves the frontend\n"
        codebase_summary += "\n**The AI can now answer detailed questions about HOW the system works internally!**\n"
        
        return codebase_summary
        
    except Exception as e:
        print(f"Error loading system codebase: {str(e)}")
        return "\n⚠️ Unable to load complete codebase information.\n"

def build_ai_prompt(user_message: str, cluster_id: Optional[int] = None) -> str:
    """
    Build a comprehensive prompt for the AI with FULL DATA ACCESS and COMPLETE PIPELINE KNOWLEDGE.
    All technical details are loaded dynamically from the actual pipeline artifacts.
    """
    
    # Load pipeline metadata from saved artifacts
    pipeline_metadata = load_pipeline_metadata()
    
    # Build system prompt dynamically from actual pipeline data
    system_prompt = f"""You are MediCluster AI, an expert medical data analyst and machine learning specialist with COMPLETE knowledge of the entire clustering pipeline.

🔬 YOUR COMPLETE TECHNICAL KNOWLEDGE (FROM ACTUAL PIPELINE):

**PIPELINE OVERVIEW:**
{pipeline_metadata.get('pipeline_description', 'Patient clustering system')}

**DATASET:**
- Total Patients: {pipeline_metadata.get('total_patients', 'N/A')}
- Total Features: {pipeline_metadata.get('num_features', 'N/A')}

"""
    
    # Add preprocessing details
    if 'preprocessing' in pipeline_metadata:
        system_prompt += "\n**PREPROCESSING PIPELINE:**\n"
        for step in pipeline_metadata['preprocessing'].get('steps', []):
            system_prompt += f"\n{step['name']}:\n"
            system_prompt += f"  - Description: {step['description']}\n"
            if 'applied_to' in step:
                system_prompt += f"  - Applied to: {', '.join(step['applied_to'])}\n"
            if 'formula' in step:
                system_prompt += f"  - Formula: {step['formula']}\n"
    
    # Add UMAP details
    if 'dimensionality_reduction' in pipeline_metadata:
        umap_info = pipeline_metadata['dimensionality_reduction']
        system_prompt += f"\n**DIMENSIONALITY REDUCTION:**\n"
        system_prompt += f"Algorithm: {umap_info.get('algorithm')} ({umap_info.get('full_name')})\n"
        system_prompt += f"Purpose: {umap_info.get('purpose')}\n"
        
        if 'parameters' in umap_info:
            system_prompt += f"\nActual Parameters Used:\n"
            for key, value in umap_info['parameters'].items():
                system_prompt += f"  - {key}: {value}\n"
        
        if 'theory' in umap_info:
            system_prompt += f"\nTheory:\n"
            for key, value in umap_info['theory'].items():
                system_prompt += f"  - {key.replace('_', ' ').title()}: {value}\n"
        
        if 'why_umap' in umap_info:
            system_prompt += f"\nWhy UMAP?\n"
            for reason in umap_info['why_umap']:
                system_prompt += f"  - {reason}\n"
    
    # Add clustering details
    if 'clustering' in pipeline_metadata:
        cluster_info = pipeline_metadata['clustering']
        system_prompt += f"\n**CLUSTERING ALGORITHM:**\n"
        system_prompt += f"Algorithm: {cluster_info.get('algorithm')}\n"
        system_prompt += f"Number of Clusters: {cluster_info.get('n_clusters')}\n"
        system_prompt += f"Description: {cluster_info.get('description')}\n"
        
        if 'kmeans_theory' in cluster_info:
            system_prompt += f"\nKMeans Theory:\n"
            for key, value in cluster_info['kmeans_theory'].items():
                system_prompt += f"  - {key.replace('_', ' ').title()}: {value}\n"
    
    # Add optimization details
    if 'optimization' in pipeline_metadata:
        opt_info = pipeline_metadata['optimization']
        system_prompt += f"\n**OPTIMIZATION PROCESS:**\n"
        system_prompt += f"Method: {opt_info.get('method')}\n"
        system_prompt += f"Optimization Metric: {opt_info.get('optimization_metric')}\n"
        system_prompt += f"Number of Trials: {opt_info.get('n_trials')}\n"
        
        if 'search_space' in opt_info:
            system_prompt += f"\nSearch Space:\n"
            for key, value in opt_info['search_space'].items():
                system_prompt += f"  - {key}: {value}\n"
        
        if 'best_parameters' in opt_info:
            system_prompt += f"\nBest Parameters Found:\n"
            for key, value in opt_info['best_parameters'].items():
                system_prompt += f"  - {key}: {value}\n"
    
    # Add evaluation metrics (THE ACTUAL RESULTS!)
    if 'evaluation_metrics' in pipeline_metadata:
        system_prompt += f"\n**ACTUAL CLUSTERING QUALITY METRICS:**\n"
        
        if 'silhouette_score' in pipeline_metadata['evaluation_metrics']:
            sil_info = pipeline_metadata['evaluation_metrics']['silhouette_score']
            system_prompt += f"\nSilhouette Score: {sil_info.get('value')} ({sil_info.get('result_interpretation')})\n"
            system_prompt += f"  - Formula: {sil_info.get('formula')}\n"
            system_prompt += f"  - Description: {sil_info.get('description')}\n"
            system_prompt += f"  - Range: {sil_info.get('range')}\n"
            system_prompt += f"  - Interpretation: {sil_info.get('interpretation')}\n"
        
        if 'davies_bouldin_index' in pipeline_metadata['evaluation_metrics']:
            db_info = pipeline_metadata['evaluation_metrics']['davies_bouldin_index']
            system_prompt += f"\nDavies-Bouldin Index: {db_info.get('value')}\n"
            system_prompt += f"  - Formula: {db_info.get('formula')}\n"
            system_prompt += f"  - Description: {db_info.get('description')}\n"
            system_prompt += f"  - Interpretation: {db_info.get('interpretation')}\n"
    
    # Add cluster distribution
    if 'cluster_distribution' in pipeline_metadata:
        system_prompt += f"\n**CLUSTER DISTRIBUTION (ACTUAL):**\n"
        for cluster_id_key, cluster_info in pipeline_metadata['cluster_distribution'].items():
            system_prompt += f"  - Cluster {cluster_id_key}: {cluster_info['count']} patients ({cluster_info['percentage']}%)\n"
    
    # Add all formulas
    if 'formulas' in pipeline_metadata:
        system_prompt += f"\n**MATHEMATICAL FORMULAS USED:**\n"
        for formula_name, formula in pipeline_metadata['formulas'].items():
            system_prompt += f"  - {formula_name.replace('_', ' ').title()}: {formula}\n"
    
    # Add visualization info
    if 'visualization' in pipeline_metadata:
        vis_info = pipeline_metadata['visualization']
        system_prompt += f"\n**VISUALIZATION:**\n"
        system_prompt += f"  - Method: {vis_info.get('method')}\n"
        system_prompt += f"  - Coordinates: {vis_info.get('coordinates')}\n"
        system_prompt += f"  - Coloring: {vis_info.get('coloring')}\n"
    
    # Add prediction pipeline
    if 'prediction_pipeline' in pipeline_metadata:
        system_prompt += f"\n**PREDICTION PIPELINE FOR NEW PATIENTS:**\n"
        for i, step in enumerate(pipeline_metadata['prediction_pipeline'], 1):
            system_prompt += f"  {i}. {step}\n"
    
    # Add complete codebase knowledge
    codebase_info = load_system_codebase()
    system_prompt += codebase_info
    
    system_prompt += """
**CRITICAL RESPONSE RULES (FOLLOW STRICTLY):**

📏 **LENGTH CONTROL:**
• Keep responses SHORT (2-4 sentences for simple questions, max 1 paragraph for complex ones)
• Answer ONLY what was asked - don't volunteer extra information
• Break up long explanations across multiple user turns instead of one long response
• If topic needs more detail, offer: "Would you like me to explain [specific aspect] further?"

❓ **WHEN UNSURE - ALWAYS ASK:**
• If question is ambiguous → Ask for clarification: "Are you asking about [A] or [B]?"
• If multiple interpretations → Ask: "Do you mean [specific detail]?"
• If missing context → Ask: "For which cluster/patient are you asking?"
• Better to ask than assume or hallucinate!

✅ **WHAT TO DO:**
• Answer the EXACT question asked
• Use specific numbers from data above (e.g., "Silhouette: 0.9192")
• One concept at a time
• If user wants more, they'll ask

❌ **WHAT NOT TO DO:**
• Don't write long paragraphs or essays
• Don't explain things not asked about
• Don't assume user's intent - ask if unclear
• Don't list all possibilities - ask which one they want
• Don't give full technical deep-dives unless specifically requested

**EXAMPLES OF GOOD RESPONSES:**

User: "What's the silhouette score?"
You: "The silhouette score is 0.9192, which indicates excellent cluster separation. Would you like to know what this metric measures?"

User: "How does clustering work?"
You: "Are you asking about (1) the algorithms we use, (2) how patients are assigned to clusters, or (3) something else?"

User: "Tell me about the clusters"
You: "We have 2 clusters: Cluster 0 (63.1%, lower risk) and Cluster 1 (36.9%, higher risk). Which cluster would you like to know more about?"

User: "How is preprocessing done?"
You: "We use 3 steps: frequency encoding for categorical variables, median imputation for missing values, and StandardScaler for normalization. Need details on any step?"

User: "What optimization method do you use?"
You: "We use Optuna with TPE sampler to optimize UMAP parameters and cluster count together over 50 trials. Want to know what parameters we optimized?"

**CONVERSATION STYLE:**
• Friendly and professional (like a helpful medical data assistant)
• Concise and focused
• Ask clarifying questions when needed
• Let the user guide the depth of explanation

Now, here is your REAL DATA ACCESS for this specific instance:
"""
    
    # === COMPREHENSIVE DATA CONTEXT ===
    data_context = "\n\n=== YOUR COMPLETE KNOWLEDGE BASE ===\n"
    
    # Check if user is asking about a specific patient
    # Look for patterns like "patient 1", "patient ID 5", "patient #10", etc.
    import re
    patient_id_match = re.search(r'patient\s*(?:id\s*)?#?(\d+)', user_message.lower())
    if patient_id_match:
        try:
            asked_patient_id = int(patient_id_match.group(1))
            patient_info = get_patient_by_id(asked_patient_id)
            
            if patient_info:
                data_context += f"\n🔍 SPECIFIC PATIENT INFORMATION (Patient ID {asked_patient_id}):\n"
                data_context += f"The user asked about Patient {asked_patient_id}. Here are their details:\n"
                data_context += f"- Cluster: {patient_info.get('cluster', 'Unknown')}\n"
                data_context += f"- Age: {patient_info.get('age', 'N/A')}\n"
                data_context += f"- Gender: {patient_info.get('gender', 'N/A')}\n"
                data_context += f"- BMI: {patient_info.get('BMI', 'N/A')}\n"
                data_context += f"- Blood Pressure: {patient_info.get('systolic_bp', 'N/A')}/{patient_info.get('diastolic_bp', 'N/A')} mmHg\n"
                data_context += f"- Diabetes: {'Yes' if patient_info.get('diabetes') == 1 else 'No'}\n"
                data_context += f"- Hypertension: {'Yes' if patient_info.get('hypertension') == 1 else 'No'}\n"
                data_context += f"- Heart Disease: {'Yes' if patient_info.get('heart_disease') == 1 else 'No'}\n"
                data_context += "\nUse this information to answer the user's question about this specific patient.\n"
            else:
                data_context += f"\n⚠️ Patient ID {asked_patient_id} not found in dataset (valid range: 1-3000).\n"
        except Exception as e:
            print(f"Error fetching patient info: {str(e)}")
    
    # 1. Dataset Overview
    try:
        dataset_stats = get_dataset_statistics()
        
        # Check if stats loaded successfully
        if 'error' in dataset_stats:
            data_context += f"\n⚠️ Dataset Overview: {dataset_stats['error']}\n"
        else:
            data_context += f"\n📊 DATASET OVERVIEW:\n"
            data_context += f"- Total Patients: {dataset_stats.get('total_patients', 0)}\n"
            
            if dataset_stats.get('cluster_distribution'):
                data_context += f"- Cluster Distribution: {dataset_stats['cluster_distribution']}\n"
            
            age_stats = dataset_stats.get('age', {})
            if age_stats and age_stats.get('mean', 0) > 0:
                data_context += f"- Age Range: {age_stats['min']:.0f}-{age_stats['max']:.0f} years (avg: {age_stats['mean']:.1f})\n"
            
            if dataset_stats.get('gender_distribution'):
                data_context += f"- Gender Distribution: {dataset_stats['gender_distribution']}\n"
            
            if dataset_stats.get('diabetes_rate', 0) > 0:
                data_context += f"- Diabetes Rate: {dataset_stats['diabetes_rate']:.1f}%\n"
            
            if dataset_stats.get('hypertension_rate', 0) > 0:
                data_context += f"- Hypertension Rate: {dataset_stats['hypertension_rate']:.1f}%\n"
            
            if dataset_stats.get('heart_disease_rate', 0) > 0:
                data_context += f"- Heart Disease Rate: {dataset_stats['heart_disease_rate']:.1f}%\n"
            
            if dataset_stats.get('average_bmi', 0) > 0:
                data_context += f"- Average BMI: {dataset_stats['average_bmi']:.1f}\n"
            
            if dataset_stats.get('average_medications', 0) > 0:
                data_context += f"- Average Medications: {dataset_stats['average_medications']:.1f}\n"
    
    except Exception as e:
        data_context += f"\n⚠️ Error loading dataset statistics: {str(e)}\n"
        print(f"Error in build_ai_prompt dataset stats: {str(e)}")
    
    # 2. Detailed Cluster Profiles
    if CLUSTER_PROFILES:
        data_context += "\n\n🔬 DETAILED CLUSTER ANALYSIS:\n"
        
        try:
            for cluster_key in sorted(CLUSTER_PROFILES.keys()):
                cluster = CLUSTER_PROFILES[cluster_key]
                cluster_num = cluster_key.split('_')[1]
                
                data_context += f"\n--- CLUSTER {cluster_num} ({cluster.get('total_patients', 'N/A')} patients, {cluster.get('percentage', 0):.1f}%) ---\n"
                
                # Risk assessment (safe access)
                risk_info = cluster.get('risk_assessment', {})
                if risk_info:
                    data_context += f"Risk Level: {risk_info.get('risk_level', 'Unknown')} (Score: {risk_info.get('risk_score', 0):.1f})\n"
                
                # Demographics (safe access)
                demo = cluster.get('demographics', {})
                if demo:
                    data_context += f"\nDemographics:\n"
                    if 'age' in demo:
                        data_context += f"  - Age: {demo['age'].get('mean', 0):.1f} ± {demo['age'].get('std', 0):.1f} years\n"
                    if 'gender' in demo:
                        data_context += f"  - Gender: Male {demo['gender'].get('Male', 0):.1f}%, Female {demo['gender'].get('Female', 0):.1f}%\n"
                    if 'ethnicity' in demo:
                        data_context += f"  - Top Ethnicity: {demo['ethnicity'].get('most_common', 'N/A')}\n"
                
                # Vital Signs (safe access)
                vitals = cluster.get('vital_signs', {})
                if vitals:
                    data_context += f"\nVital Signs:\n"
                    if 'BMI' in vitals:
                        data_context += f"  - BMI: {vitals['BMI'].get('mean', 0):.1f} ({vitals['BMI'].get('category', 'N/A')})\n"
                    if 'systolic_bp' in vitals and 'diastolic_bp' in vitals:
                        data_context += f"  - Blood Pressure: {vitals['systolic_bp'].get('mean', 0):.0f}/{vitals['diastolic_bp'].get('mean', 0):.0f} mmHg ({vitals.get('blood_pressure_category', 'N/A')})\n"
                    if 'heart_rate' in vitals:
                        data_context += f"  - Heart Rate: {vitals['heart_rate'].get('mean', 0):.0f} bpm\n"
                    if 'cholesterol_total' in vitals:
                        data_context += f"  - Cholesterol: {vitals['cholesterol_total'].get('mean', 0):.0f} mg/dL ({vitals.get('cholesterol_category', 'N/A')})\n"
                
                # Conditions (safe access)
                cond = cluster.get('health_conditions', {})
                if cond:
                    data_context += f"\nHealth Conditions:\n"
                    if 'diabetes' in cond:
                        data_context += f"  - Diabetes: {cond['diabetes'].get('percentage', 0):.1f}%\n"
                    if 'hypertension' in cond:
                        data_context += f"  - Hypertension: {cond['hypertension'].get('percentage', 0):.1f}%\n"
                    if 'heart_disease' in cond:
                        data_context += f"  - Heart Disease: {cond['heart_disease'].get('percentage', 0):.1f}%\n"
                
                # Lifestyle (safe access)
                lifestyle = cluster.get('lifestyle_factors', {})
                if lifestyle:
                    data_context += f"\nLifestyle:\n"
                    if 'smoking_status' in lifestyle:
                        data_context += f"  - Never Smokers: {lifestyle['smoking_status'].get('Never', 0):.1f}%\n"
                    if 'alcohol_consumption' in lifestyle:
                        data_context += f"  - Alcohol: Moderate {lifestyle['alcohol_consumption'].get('Moderate', 0):.1f}%\n"
                
                # Treatment (safe access)
                treatment = cluster.get('treatment_metrics', {})
                if treatment:
                    data_context += f"\nTreatment:\n"
                    if 'num_medications' in treatment:
                        data_context += f"  - Avg Medications: {treatment['num_medications'].get('mean', 0):.1f}\n"
                    if 'medication_adherence' in treatment:
                        data_context += f"  - Medication Adherence: {treatment['medication_adherence'].get('mean', 0)*100:.1f}%\n"
                    if 'treatment_success_rate' in treatment:
                        data_context += f"  - Treatment Success: {treatment['treatment_success_rate'].get('mean', 0)*100:.1f}%\n"
                    if 'doctor_visits_per_year' in treatment:
                        data_context += f"  - Doctor Visits/Year: {treatment['doctor_visits_per_year'].get('mean', 0):.1f}\n"
                
                # Key Characteristics (safe access)
                if 'key_characteristics' in cluster:
                    data_context += f"\nKey Characteristics: {', '.join(cluster['key_characteristics'])}\n"
                    
        except Exception as e:
            data_context += f"\n⚠️ Error loading cluster profiles: {str(e)}\n"
            print(f"Error in build_ai_prompt cluster profiles: {str(e)}")
    
    # 3. If specific cluster requested, add focused data
    if cluster_id is not None:
        try:
            data_context += f"\n\n🎯 FOCUSED ANALYSIS - CLUSTER {cluster_id}:\n"
            data_context += "The user is specifically asking about this cluster. Provide detailed, cluster-specific insights.\n"
            
            # Get sample patients from this cluster (anonymized)
            sample_patients = search_patients_by_criteria({'cluster': cluster_id})
            if sample_patients:
                data_context += f"\nSample Patient Patterns in Cluster {cluster_id}:\n"
                for i, patient in enumerate(sample_patients[:3], 1):
                    age = patient.get('age', 'N/A')
                    bmi = patient.get('BMI')
                    bmi_str = f"{bmi:.1f}" if isinstance(bmi, (int, float)) else 'N/A'
                    diabetes = 'Yes' if patient.get('diabetes') == 1 else 'No'
                    hypertension = 'Yes' if patient.get('hypertension') == 1 else 'No'
                    data_context += f"  Example {i}: Age {age}, BMI {bmi_str}, Diabetes: {diabetes}, Hypertension: {hypertension}\n"
        except Exception as e:
            data_context += f"\n⚠️ Error loading sample patients: {str(e)}\n"
            print(f"Error in build_ai_prompt sample patients: {str(e)}")
    
    # 4. Model Performance Metrics (if available)
    try:
        # Try to load clustering quality metrics
        metrics_path = os.path.join(config.ARTIFACTS_DIR, "clustering_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            data_context += "\n\n📈 CLUSTERING QUALITY METRICS:\n"
            if 'silhouette_score' in metrics:
                data_context += f"- Silhouette Score: {metrics['silhouette_score']:.4f} (measures cluster separation quality, -1 to 1, higher is better)\n"
            if 'calinski_harabasz_score' in metrics:
                data_context += f"- Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f} (cluster density/separation ratio)\n"
            if 'davies_bouldin_score' in metrics:
                data_context += f"- Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (average similarity between clusters, lower is better)\n"
            if 'num_clusters' in metrics:
                data_context += f"- Number of Clusters: {metrics['num_clusters']}\n"
            if 'umap_dimensions' in metrics:
                data_context += f"- UMAP Dimensions: {metrics['umap_dimensions']} (for clustering) + 2D (for visualization)\n"
    except Exception as e:
        # Metrics file might not exist, that's okay
        pass
    
    # 5. Available data fields (so AI knows what it can analyze)
    data_context += "\n\n📋 AVAILABLE DATA FIELDS FOR EACH PATIENT:\n"
    data_context += """- patient_id (1-3000), age, gender, ethnicity, insurance_type
- BMI, systolic_bp, diastolic_bp, heart_rate
- cholesterol_total, blood_glucose
- diabetes (0/1), hypertension (0/1), heart_disease (0/1)
- smoking_status, alcohol_consumption
- doctor_visits_per_year, num_medications
- medication_adherence, treatment_success_rate
- cluster (0 or 1), UMAP coordinates (x, y)

💡 IMPORTANT - PATIENT LOOKUPS:
• You CAN access specific patient data when asked (e.g., "What cluster is patient 1 in?")
• If patient information is provided above in "SPECIFIC PATIENT INFORMATION", use it to answer
• If patient is mentioned but info not provided, the system will try to fetch it
• Always include the cluster assignment when discussing a specific patient

NOTE: All statistics above are REAL data from the actual analysis. Use these exact numbers in your responses.
"""
    
    # User's actual question
    user_query = f"\n\n=== USER QUESTION ===\n{user_message}\n"
    user_query += "\n⚠️ CRITICAL INSTRUCTIONS:\n"
    user_query += "1. Keep your response SHORT (2-4 sentences max unless explicitly asked for detail)\n"
    user_query += "2. Answer ONLY what was asked - don't add extra information\n"
    user_query += "3. If the question is unclear or ambiguous, ASK for clarification instead of guessing\n"
    user_query += "4. Use REAL numbers from the data above when answering\n"
    user_query += "5. If you need to explain something complex, break it into simple chunks and ask if user wants more\n"
    user_query += "6. NEVER write long paragraphs or essays - be concise and focused\n\n"
    
    # Combine everything
    full_prompt = system_prompt + data_context + user_query
    
    return full_prompt

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting Flask API server...")
    print(f"API will be available at: http://0.0.0.0:{port}")
    print("Endpoints:")
    print("  - POST /predict - Predict cluster for new patient")
    print("  - GET  /api/get_all_patients - Get visualization data")
    print("  - GET  /api/get_patient/<id> - Get full patient details")
    print("  - POST /api/chat - AI chatbot for cluster insights")
    print("  - GET  /health - Health check")
    app.run(debug=False, host='0.0.0.0', port=port)
