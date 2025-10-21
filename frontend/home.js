// API Configuration
// API Configuration - Use config from config.js
const API_BASE_URL = API_CONFIG.BACKEND_URL;

// Global variables
let allPatientsData = [];

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
    console.log('🚀 ClusterMed Home Page initialized');
    
    // Check API health
    await checkAPIHealth();
    
    // Load all patients data for visualization
    await loadAllPatientsData();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update stats
    updateStats();
    
    // Render initial scatter plot
    renderScatterPlot();
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
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'API Offline';
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
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
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
    document.getElementById('totalClusters').textContent = uniqueClusters.size || '-';
    
    // Total patients
    document.getElementById('totalPatients').textContent = 
        allPatientsData.length > 0 ? allPatientsData.length.toLocaleString() : '-';
    
    // Model status
    document.getElementById('modelStatus').textContent = 
        allPatientsData.length > 0 ? 'Ready' : 'Training Required';
    
    // Update cluster recommendation counts (if elements exist)
    const cluster0Count = allPatientsData.filter(p => p.cluster === 0).length;
    const cluster1Count = allPatientsData.filter(p => p.cluster === 1).length;
    
    const cluster0Elem = document.getElementById('cluster0Count');
    const cluster1Elem = document.getElementById('cluster1Count');
    
    if (cluster0Elem) {
        cluster0Elem.textContent = `${cluster0Count.toLocaleString()} patients`;
    }
    if (cluster1Elem) {
        cluster1Elem.textContent = `${cluster1Count.toLocaleString()} patients`;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Search button
    document.getElementById('searchBtn').addEventListener('click', handleSearch);
    
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', handleRefresh);
    
    // Enter key on search input
    document.getElementById('patientSearchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Cluster explanation toggle for dashboard
    const dashboardExplanationToggle = document.getElementById('dashboardClusterExplanationToggle');
    const dashboardExplanationContent = document.getElementById('dashboardClusterExplanationContent');
    
    if (dashboardExplanationToggle) {
        dashboardExplanationToggle.addEventListener('click', () => {
            dashboardExplanationToggle.classList.toggle('active');
            dashboardExplanationContent.classList.toggle('show');
        });
    }
}

// Handle refresh data
async function handleRefresh() {
    const refreshBtn = document.getElementById('refreshBtn');
    const originalText = refreshBtn.innerHTML;
    
    try {
        // Disable button and show loading state
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<span class="btn-icon">⏳</span> Refreshing...';
        
        // Reload patient data
        await loadAllPatientsData();
        
        // Update stats
        updateStats();
        
        // Re-render scatter plot
        renderScatterPlot();
        
        showNotification(`Data refreshed! Total patients: ${allPatientsData.length}`, 'success');
        
    } catch (error) {
        console.error('Refresh error:', error);
        showNotification('Failed to refresh data', 'error');
    } finally {
        // Re-enable button
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = originalText;
    }
}

// Handle patient search
function handleSearch() {
    const searchInput = document.getElementById('patientSearchInput');
    const searchValue = searchInput.value.trim();
    
    if (!searchValue) {
        showNotification('Please enter a Patient ID', 'warning');
        return;
    }
    
    // Validate that input is numeric
    if (!/^\d+$/.test(searchValue)) {
        showNotification('Patient ID must be a number', 'error');
        return;
    }
    
    // Convert search value to number for comparison
    const searchId = parseInt(searchValue);
    
    // Search for patient in loaded data
    const patient = allPatientsData.find(p => p.patient_id === searchId);
    
    if (patient) {
        displaySearchResults(patient);
    } else {
        showNotification(`Patient ID "${searchValue}" not found in the dataset`, 'error');
        document.getElementById('searchResults').style.display = 'none';
        document.getElementById('patientDetailsCard').style.display = 'none';
    }
}

// Display search results
async function displaySearchResults(patient) {
    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.style.display = 'block';
    
    // Helper function to safely display values
    const safeValue = (value) => (value !== null && value !== undefined && value !== 'N/A') ? value : 'N/A';
    
    // Update result values - Basic info in results grid
    document.getElementById('searchPatientId').textContent = patient.patient_id || 'N/A';
    document.getElementById('searchCluster').textContent = `Cluster ${patient.cluster}`;
    document.getElementById('searchCluster').className = `result-value cluster-badge cluster-${patient.cluster}`;
    
    // Fetch full patient details to populate all fields
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_patient/${patient.patient_id}`);
        if (response.ok) {
            const result = await response.json();
            if (result.success && result.patient) {
                const fullPatient = result.patient;
                
                // Update all search result fields
                document.getElementById('searchAge').textContent = safeValue(fullPatient.age);
                document.getElementById('searchGender').textContent = safeValue(fullPatient.gender);
                
                const bmi = safeValue(fullPatient.BMI);
                document.getElementById('searchBMI').textContent = (bmi !== 'N/A') ? Number(bmi).toFixed(1) : 'N/A';
                
                const systolic = safeValue(fullPatient.systolic_bp);
                const diastolic = safeValue(fullPatient.diastolic_bp);
                document.getElementById('searchBP').textContent = (systolic !== 'N/A' && diastolic !== 'N/A')
                    ? `${systolic}/${diastolic}` : 'N/A';
                
                const diabetes = safeValue(fullPatient.diabetes);
                document.getElementById('searchDiabetes').textContent = diabetes === 1 ? 'Yes' : diabetes === 0 ? 'No' : 'N/A';
                
                const hypertension = safeValue(fullPatient.hypertension);
                document.getElementById('searchHypertension').textContent = hypertension === 1 ? 'Yes' : hypertension === 0 ? 'No' : 'N/A';
                
                // Additional health metrics
                const heartRate = safeValue(fullPatient.heart_rate);
                document.getElementById('searchHeartRate').textContent = (heartRate !== 'N/A') ? `${heartRate} bpm` : 'N/A';
                
                const cholesterol = safeValue(fullPatient.cholesterol_total);
                document.getElementById('searchCholesterol').textContent = (cholesterol !== 'N/A') ? `${cholesterol} mg/dL` : 'N/A';
                
                const glucose = safeValue(fullPatient.blood_glucose);
                document.getElementById('searchGlucose').textContent = (glucose !== 'N/A') ? `${glucose} mg/dL` : 'N/A';
                
                document.getElementById('searchMedications').textContent = safeValue(fullPatient.num_medications);
                document.getElementById('searchVisits').textContent = safeValue(fullPatient.doctor_visits_per_year);
                
                const success = safeValue(fullPatient.treatment_success_rate);
                document.getElementById('searchSuccess').textContent = (success !== 'N/A')
                    ? `${(success * 100).toFixed(0)}%` : 'N/A';
                
                // Update cluster recommendation
                const recDiv = document.getElementById('clusterRec');
                const recText = document.getElementById('recText');
                if (recDiv && recText) {
                    recDiv.style.display = 'flex';
                    if (fullPatient.cluster === 0) {
                        recText.textContent = 'Low-risk profile. Recommend routine monitoring and preventive care.';
                    } else if (fullPatient.cluster === 1) {
                        recText.textContent = 'High-risk profile. Recommend intensive monitoring, medication adjustment, and lifestyle interventions.';
                    }
                }
                
                // Populate detailed cluster explanation
                populateDashboardClusterExplanation(fullPatient.cluster);
            }
        }
    } catch (error) {
        console.error('Error loading full patient details:', error);
    }
    
    // Highlight patient on scatter plot
    renderScatterPlot(patient);
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
    
    showNotification('Patient found successfully!', 'success');
}

// Load full patient details
async function loadPatientDetails(patientId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_patient/${patientId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load patient details`);
        }
        
        const result = await response.json();
        
        if (result.success && result.patient) {
            displayPatientDetails(result.patient);
        }
    } catch (error) {
        console.error('Error loading patient details:', error);
        showNotification('Could not load full patient details', 'warning');
    }
}

// Display full patient details
function displayPatientDetails(patient) {
    const detailsCard = document.getElementById('patientDetailsCard');
    detailsCard.style.display = 'block';
    
    // Helper function to safely display values (handles null, undefined, 'N/A')
    const safeValue = (value) => (value !== null && value !== undefined && value !== 'N/A') ? value : 'N/A';
    
    // Update patient ID
    document.getElementById('detailsPatientId').textContent = safeValue(patient.patient_id);
    
    // Demographics
    document.getElementById('detailAge').textContent = safeValue(patient.age);
    document.getElementById('detailGender').textContent = safeValue(patient.gender);
    document.getElementById('detailEthnicity').textContent = safeValue(patient.ethnicity);
    document.getElementById('detailInsurance').textContent = safeValue(patient.insurance_type);
    
    // Vital Signs
    const bmi = safeValue(patient.BMI);
    document.getElementById('detailBMI').textContent = (bmi !== 'N/A') ? Number(bmi).toFixed(1) : 'N/A';
    
    const systolic = safeValue(patient.systolic_bp);
    const diastolic = safeValue(patient.diastolic_bp);
    document.getElementById('detailBloodPressure').textContent = (systolic !== 'N/A' && diastolic !== 'N/A')
        ? `${systolic}/${diastolic} mmHg` : 'N/A';
    
    const glucose = safeValue(patient.blood_glucose);
    document.getElementById('detailGlucose').textContent = (glucose !== 'N/A') ? `${glucose} mg/dL` : 'N/A';
    
    const cholesterol = safeValue(patient.cholesterol_total);
    document.getElementById('detailCholesterol').textContent = (cholesterol !== 'N/A') ? `${cholesterol} mg/dL` : 'N/A';
    
    // Health Conditions
    const diabetes = safeValue(patient.diabetes);
    document.getElementById('detailDiabetes').textContent = diabetes === 1 ? 'Yes' : diabetes === 0 ? 'No' : 'N/A';
    
    const hypertension = safeValue(patient.hypertension);
    document.getElementById('detailHypertension').textContent = hypertension === 1 ? 'Yes' : hypertension === 0 ? 'No' : 'N/A';
    
    const heartDisease = safeValue(patient.heart_disease);
    document.getElementById('detailHeartDisease').textContent = heartDisease === 1 ? 'Yes' : heartDisease === 0 ? 'No' : 'N/A';
    
    document.getElementById('detailSmoking').textContent = safeValue(patient.smoking_status);
    document.getElementById('detailAlcohol').textContent = safeValue(patient.alcohol_consumption);
    
    // Treatment
    document.getElementById('detailVisits').textContent = safeValue(patient.doctor_visits_per_year);
    document.getElementById('detailMedications').textContent = safeValue(patient.num_medications);
    
    const adherence = safeValue(patient.medication_adherence);
    document.getElementById('detailAdherence').textContent = (adherence !== 'N/A')
        ? `${(adherence * 100).toFixed(0)}%` : 'N/A';
    
    const success = safeValue(patient.treatment_success_rate);
    document.getElementById('detailSuccess').textContent = (success !== 'N/A')
        ? `${(success * 100).toFixed(0)}%` : 'N/A';
    
    // Show note if it's a new patient
    if (patient.note) {
        const noteElement = document.createElement('div');
        noteElement.className = 'patient-note';
        noteElement.style.cssText = 'background: rgba(79, 70, 229, 0.1); border-left: 4px solid #4F46E5; padding: 1rem; margin-top: 1rem; border-radius: 8px; color: #E5E7EB;';
        noteElement.innerHTML = `<strong>ℹ️ Note:</strong> ${patient.note}`;
        
        const detailsCard = document.getElementById('patientDetailsCard');
        const existingNote = detailsCard.querySelector('.patient-note');
        if (existingNote) existingNote.remove();
        detailsCard.appendChild(noteElement);
    }
    
    // Scroll to details
    detailsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create scatter plot with Plotly
function renderScatterPlot(highlightPatient = null) {
    if (allPatientsData.length === 0) {
        document.getElementById('clusterPlot').innerHTML = 
            '<div class="plot-error">⚠️ No data available. Please run train.py first.</div>';
        return;
    }
    
    // Group patients by cluster
    const clusterGroups = {};
    allPatientsData.forEach(patient => {
        const cluster = patient.cluster;
        if (!clusterGroups[cluster]) {
            clusterGroups[cluster] = [];
        }
        clusterGroups[cluster].push(patient);
    });
    
    // Create traces for each cluster
    const traces = [];
    Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {
        const patients = clusterGroups[cluster];
        const clusterNum = parseInt(cluster);
        
        traces.push({
            x: patients.map(p => p.x),
            y: patients.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: `Cluster ${cluster}`,
            text: patients.map(p => `Patient: ${p.patient_id || 'N/A'}<br>Cluster: ${cluster}`),
            marker: {
                size: 8,
                color: CLUSTER_COLORS[clusterNum % CLUSTER_COLORS.length],
                opacity: 0.7
            }
        });
    });
    
    // Add highlighted patient if provided
    if (highlightPatient) {
        traces.push({
            x: [highlightPatient.x],
            y: [highlightPatient.y],
            mode: 'markers',
            type: 'scatter',
            name: 'Selected Patient',
            text: [`Patient: ${highlightPatient.patient_id || 'N/A'}<br>Cluster: ${highlightPatient.cluster}`],
            marker: {
                size: 20,
                color: '#EF4444',
                symbol: 'star',
                line: {
                    color: 'white',
                    width: 2
                }
            }
        });
    }
    
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
        },
        margin: {
            l: 60,
            r: 60,
            t: 60,
            b: 60,
            pad: 5
        },
        autosize: true
    };
    
    // Plot configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create the plot - use Plotly.react for better updates
    Plotly.newPlot('clusterPlot', traces, layout, config).then(function() {
        // Ensure plot fits container after creation
        const plotDiv = document.getElementById('clusterPlot');
        if (plotDiv) {
            Plotly.Plots.resize(plotDiv);
        }
    });
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

// Populate detailed cluster explanation for dashboard
function populateDashboardClusterExplanation(cluster) {
    const explanations = {
        0: {
            characteristics: 'Patients in Cluster 0 typically have stable vital signs, minimal comorbidities, and good medication adherence. They represent a low-risk population with well-controlled health conditions, normal BMI ranges, and regular healthcare engagement patterns.',
            riskFactors: 'Low risk profile with minimal health concerns. May have occasional elevated blood pressure or blood glucose levels, but generally within manageable ranges. Lifestyle factors such as diet and exercise are typically well-maintained.',
            recommendations: 'Routine annual checkups and preventive care screenings. Maintain current medication regimen and continue healthy lifestyle practices. Focus on preventive measures such as vaccinations, health education, and wellness programs. Encourage continued engagement with primary care provider.'
        },
        1: {
            characteristics: 'Cluster 1 patients show moderate to high health complexity with one or more chronic conditions requiring ongoing management. They may have hypertension, diabetes, or elevated cholesterol levels that require regular monitoring and medication adjustments.',
            riskFactors: 'Moderate to high risk with potential for disease progression if not properly managed. Risk factors include inconsistent medication adherence, suboptimal blood pressure or glucose control, and lifestyle factors that may exacerbate existing conditions.',
            recommendations: 'Intensive monitoring with bi-annual or quarterly checkups. Medication adjustment and optimization may be necessary. Implement lifestyle interventions including dietary counseling, exercise programs, and stress management. Consider referral to disease management programs or specialists for complex cases.'
        }
    };
    
    const defaultExplanation = {
        characteristics: `Patients in Cluster ${cluster} represent a distinct subgroup within the patient population with specific health characteristics and care needs. This cluster exhibits unique patterns in vital signs, comorbidity profiles, and healthcare utilization.`,
        riskFactors: `Risk factors for this cluster include various health indicators that require careful monitoring and management. The specific risk profile is determined by analyzing multiple clinical parameters and patient demographics.`,
        recommendations: `Clinical management should be tailored to the specific needs of this patient cluster. Regular monitoring, appropriate medication management, and lifestyle interventions should be considered based on individual patient characteristics and risk factors.`
    };
    
    const explanation = explanations[cluster] || defaultExplanation;
    
    document.getElementById('dashboardClusterCharacteristics').textContent = explanation.characteristics;
    document.getElementById('dashboardClusterRiskFactors').textContent = explanation.riskFactors;
    document.getElementById('dashboardClusterRecommendations').textContent = explanation.recommendations;
}

// Resize plot on window resize for better responsiveness
window.addEventListener('resize', function() {
    const plotElement = document.getElementById('clusterPlot');
    if (plotElement && plotElement.data) {
        Plotly.Plots.resize('clusterPlot');
    }
});

// ==============================================
// Chatbot functionality is now in chatbot.js (shared component)

// Log application info
console.log('%c🔬 ClusterMed - Home Dashboard', 'color: #4F46E5; font-size: 20px; font-weight: bold;');
console.log('%cAPI Endpoint: ' + API_BASE_URL, 'color: #10B981;');
console.log('%cFeatures: Patient Search, Cluster Visualization & AI Chat', 'color: #F59E0B;');

