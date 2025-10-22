// API Configuration
const API_BASE_URL = 'API_CONFIG.BACKEND_URL';

// Global variables
let allPatientsData = [];
let currentPrediction = null;

// Color scheme for clusters
const CLUSTER_COLORS = [
    '#4F46E5', // Purple
    '#10B981', // Green
    '#F59E0B', // Orange
    '#EF4444', // Red
    '#8B5CF6', // Violet
    '#EC4899', // Pink
    '#14B8A6', // Teal
    '#F97316', // Orange-red
    '#6366F1', // Indigo
    '#A855F7'  // Purple-pink
];

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 ClusterMed initialized');
    
    // Check API health
    await checkAPIHealth();
    
    // Load all patients data for visualization
    await loadAllPatientsData();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update stats
    updateStats();
});

// Check if API is running
async function checkAPIHealth() {
    const statusIndicator = document.getElementById('apiStatus');
    const statusText = document.getElementById('apiStatusText');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'API Online';
            console.log('✅ API is healthy');
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'API Offline';
        console.error('❌ API health check failed:', error);
        showNotification('API server is not running. Please start the Flask server.', 'error');
    }
}

// Load all patients data from backend
async function loadAllPatientsData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_all_patients`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        allPatientsData = await response.json();
        console.log(`✅ Loaded ${allPatientsData.length} patients for visualization`);
        
    } catch (error) {
        console.error('❌ Failed to load patients data:', error);
        showNotification('Failed to load visualization data. Make sure train.py has been run.', 'warning');
        allPatientsData = [];
    }
}

// Update statistics
function updateStats() {
    // Count unique clusters
    const uniqueClusters = new Set(allPatientsData.map(p => p.cluster));
    
    // Update elements only if they exist (not all pages have these elements)
    const totalClustersElem = document.getElementById('totalClusters');
    const totalPatientsElem = document.getElementById('totalPatients');
    const modelStatusElem = document.getElementById('modelStatus');
    
    if (totalClustersElem) {
        totalClustersElem.textContent = uniqueClusters.size || '-';
    }
    
    if (totalPatientsElem) {
        totalPatientsElem.textContent = allPatientsData.length > 0 ? allPatientsData.length.toLocaleString() : '-';
    }
    
    if (modelStatusElem) {
        modelStatusElem.textContent = allPatientsData.length > 0 ? 'Ready' : 'Training Required';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    document.getElementById('patientForm').addEventListener('submit', handleFormSubmit);
    
    // Fill sample data button
    document.getElementById('fillSampleBtn').addEventListener('click', fillSampleData);
    
    // Cluster explanation toggle
    const explanationToggle = document.getElementById('clusterExplanationToggle');
    const explanationContent = document.getElementById('clusterExplanationContent');
    
    if (explanationToggle) {
        explanationToggle.addEventListener('click', () => {
            explanationToggle.classList.toggle('active');
            explanationContent.classList.toggle('show');
        });
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.innerHTML;
    
    // Show loading state
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Predicting...';
    
    // Collect form data
    const formData = new FormData(event.target);
    const patientData = {};
    
    for (let [key, value] of formData.entries()) {
        // Convert numeric fields to numbers
        if (['age', 'BMI', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
             'cholesterol_total', 'blood_glucose', 'diabetes', 'hypertension', 
             'heart_disease', 'doctor_visits_per_year', 'num_medications', 
             'medication_adherence', 'treatment_success_rate'].includes(key)) {
            patientData[key] = parseFloat(value);
        } else {
            patientData[key] = value;
        }
    }
    
    console.log('📤 Sending prediction request:', patientData);
    
    try {
        // Call prediction API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Prediction result:', result);
        
        if (result.success) {
            currentPrediction = result;
            displayResults(result);
            showNotification(`Patient saved with ID ${result.patient_id}. Total patients: ${result.patient_id + 1}`, 'success');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        showNotification(`Prediction failed: ${error.message}`, 'error');
    } finally {
        // Reset button
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    }
}

// Display prediction results
function displayResults(result) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display patient ID
    document.getElementById('assignedPatientId').textContent = result.patient_id;
    
    // Display cluster number
    document.getElementById('predictedCluster').textContent = `Cluster ${result.cluster}`;
    document.getElementById('predictedCluster').style.color = CLUSTER_COLORS[result.cluster % CLUSTER_COLORS.length];
    
    // Display cluster description
    document.getElementById('clusterDescription').textContent = getClusterDescription(result.cluster);
    
    // Populate detailed cluster explanation
    populateClusterExplanation(result.cluster);
    
    // Display UMAP coordinates
    document.getElementById('umapX').textContent = result.umap_coordinates.x.toFixed(3);
    document.getElementById('umapY').textContent = result.umap_coordinates.y.toFixed(3);
    
    // Create scatter plot
    createScatterPlot(result);
}

// Get cluster description
function getClusterDescription(cluster) {
    const descriptions = {
        0: 'Low risk patient group with stable health indicators',
        1: 'Moderate risk group requiring regular monitoring',
        2: 'High risk group with multiple comorbidities',
        3: 'Elderly care group with age-related conditions',
        4: 'Young adult group with preventive care needs',
        5: 'Chronic disease management group',
        6: 'Acute care group requiring immediate attention',
        7: 'Recovery and rehabilitation group',
        8: 'Preventive care and wellness group',
        9: 'Complex care coordination group'
    };
    
    return descriptions[cluster] || `Patient group ${cluster} - specialized care profile`;
}

// Populate detailed cluster explanation
function populateClusterExplanation(cluster) {
    const explanations = {
        0: {
            characteristics: 'Patients in Cluster 0 typically have stable vital signs, minimal comorbidities, and good medication adherence. They represent a low-risk population with well-controlled health conditions, normal BMI ranges, and regular healthcare engagement patterns.',
            riskFactors: 'Low risk profile with minimal health concerns. May have occasional elevated blood pressure or blood glucose levels, but generally within manageable ranges. Lifestyle factors such as diet and exercise are typically well-maintained.',
            recommendations: 'Routine annual checkups and preventive care screenings. Maintain current medication regimen and continue healthy lifestyle practices. Focus on preventive measures such as vaccinations, health education, and wellness programs. Encourage continued engagement with primary care provider.'
        },
        1: {
            characteristics: 'Cluster 1 patients show moderate health complexity with one or more chronic conditions requiring ongoing management. They may have hypertension, diabetes, or elevated cholesterol levels that require regular monitoring and medication adjustments.',
            riskFactors: 'Moderate risk with potential for disease progression if not properly managed. Risk factors include inconsistent medication adherence, suboptimal blood pressure or glucose control, and lifestyle factors that may exacerbate existing conditions.',
            recommendations: 'Intensive monitoring with bi-annual or quarterly checkups. Medication adjustment and optimization may be necessary. Implement lifestyle interventions including dietary counseling, exercise programs, and stress management. Consider referral to disease management programs or specialists for complex cases.'
        }
    };
    
    const defaultExplanation = {
        characteristics: `Patients in Cluster ${cluster} represent a distinct subgroup within the patient population with specific health characteristics and care needs. This cluster exhibits unique patterns in vital signs, comorbidity profiles, and healthcare utilization.`,
        riskFactors: `Risk factors for this cluster include various health indicators that require careful monitoring and management. The specific risk profile is determined by analyzing multiple clinical parameters and patient demographics.`,
        recommendations: `Clinical management should be tailored to the specific needs of this patient cluster. Regular monitoring, appropriate medication management, and lifestyle interventions should be considered based on individual patient characteristics and risk factors.`
    };
    
    const explanation = explanations[cluster] || defaultExplanation;
    
    document.getElementById('clusterCharacteristics').textContent = explanation.characteristics;
    document.getElementById('clusterRiskFactors').textContent = explanation.riskFactors;
    document.getElementById('clusterRecommendations').textContent = explanation.recommendations;
}

// Create scatter plot with Plotly
function createScatterPlot(prediction) {
    if (allPatientsData.length === 0) {
        document.getElementById('scatterPlot').innerHTML = 
            '<div class="plot-error">⚠️ Visualization data not available. Please run train.py first.</div>';
        return;
    }
    
    // Group patients by cluster
    const clusterGroups = {};
    allPatientsData.forEach(patient => {
        if (!clusterGroups[patient.cluster]) {
            clusterGroups[patient.cluster] = { x: [], y: [] };
        }
        clusterGroups[patient.cluster].x.push(patient.x);
        clusterGroups[patient.cluster].y.push(patient.y);
    });
    
    // Create traces for each cluster
    const traces = [];
    Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {
        const clusterNum = parseInt(cluster);
        traces.push({
            x: clusterGroups[cluster].x,
            y: clusterGroups[cluster].y,
            mode: 'markers',
            type: 'scatter',
            name: `Cluster ${cluster}`,
            marker: {
                size: 8,
                color: CLUSTER_COLORS[clusterNum % CLUSTER_COLORS.length],
                opacity: 0.6,
                line: {
                    color: 'white',
                    width: 0.5
                }
            }
        });
    });
    
    // Add the new patient as a highlighted point
    traces.push({
        x: [prediction.umap_coordinates.x],
        y: [prediction.umap_coordinates.y],
        mode: 'markers',
        type: 'scatter',
        name: 'Your Patient',
        marker: {
            size: 20,
            color: 'rgba(239, 68, 68, 0.5)',  // Red with 50% opacity
            symbol: 'star',
            line: {
                color: 'white',
                width: 2
            }
        }
    });
    
    // Layout configuration
    const layout = {
        title: {
            text: 'Patient Cluster Distribution (UMAP Projection)',
            font: { size: 18, color: '#E5E7EB' }
        },
        xaxis: {
            title: 'UMAP Dimension 1',
            gridcolor: '#374151',
            zerolinecolor: '#4B5563',
            color: '#9CA3AF'
        },
        yaxis: {
            title: 'UMAP Dimension 2',
            gridcolor: '#374151',
            zerolinecolor: '#4B5563',
            color: '#9CA3AF'
        },
        plot_bgcolor: '#1F2937',
        paper_bgcolor: '#111827',
        font: {
            color: '#E5E7EB'
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1.02,
            y: 1,
            bgcolor: '#1F2937',
            bordercolor: '#374151',
            borderwidth: 1
        }
    };
    
    // Plot configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create the plot
    Plotly.newPlot('scatterPlot', traces, layout, config);
}

// Fill sample data for testing
function fillSampleData() {
    document.getElementById('age').value = 55;
    document.getElementById('gender').value = 'Male';
    document.getElementById('ethnicity').value = 'Hispanic';
    document.getElementById('insurance_type').value = 'Private';
    document.getElementById('BMI').value = 28.0;
    document.getElementById('systolic_bp').value = 135;
    document.getElementById('diastolic_bp').value = 85;
    document.getElementById('heart_rate').value = 75;
    document.getElementById('cholesterol_total').value = 210;
    document.getElementById('blood_glucose').value = 110;
    document.getElementById('diabetes').value = '0';
    document.getElementById('hypertension').value = '1';
    document.getElementById('heart_disease').value = '0';
    document.getElementById('smoking_status').value = 'Never';
    document.getElementById('alcohol_consumption').value = 'Moderate';
    document.getElementById('doctor_visits_per_year').value = 3;
    document.getElementById('num_medications').value = 2;
    document.getElementById('medication_adherence').value = 0.9;
    document.getElementById('treatment_success_rate').value = 0.85;
    
    showNotification('Sample data filled successfully!', 'info');
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotif = document.querySelector('.notification');
    if (existingNotif) {
        existingNotif.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icon = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }[type] || 'ℹ️';
    
    notification.innerHTML = `
        <span class="notification-icon">${icon}</span>
        <span class="notification-message">${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Log application info
console.log('%c🔬 ClusterMed - Patient Clustering System', 'color: #4F46E5; font-size: 20px; font-weight: bold;');
console.log('%cAPI Endpoint: ' + API_BASE_URL, 'color: #10B981;');
console.log('%cMake sure your Flask server is running on port 5000', 'color: #F59E0B;');
