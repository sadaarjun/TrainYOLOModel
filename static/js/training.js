/**
 * JavaScript file specifically for training YOLO models
 */

// Global variables
let trainingId = null;
let trainingStatus = null;
let statusCheckInterval = null;
let trainingChart = null;

// Initialize training functionality
document.addEventListener('DOMContentLoaded', function() {
    const startTrainingBtn = document.getElementById('start-training-btn');
    if (startTrainingBtn) {
        startTrainingBtn.addEventListener('click', startTraining);
    }
    
    // Initialize Chart.js if the element exists
    const chartCanvas = document.getElementById('training-metrics-chart');
    if (chartCanvas) {
        initTrainingChart(chartCanvas);
    }
});

/**
 * Initialize the training metrics chart
 */
function initTrainingChart(canvas) {
    const ctx = canvas.getContext('2d');
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'mAP50',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'mAP50-95',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Precision',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Recall',
                    data: [],
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Metrics'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

/**
 * Start the training process
 */
function startTraining() {
    // Get training parameters from the form
    const datasetPath = document.getElementById('dataset-path').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const imgSize = parseInt(document.getElementById('img-size').value);
    const modelType = document.getElementById('model-type').value;
    const pretrained = document.getElementById('pretrained').checked;
    
    if (!datasetPath) {
        showAlert('Please upload or create a dataset first', 'warning');
        return;
    }
    
    // Show loading indicator
    showLoading('Starting training...');
    document.getElementById('training-interface').classList.add('d-none');
    document.getElementById('training-progress').classList.remove('d-none');
    
    // Reset training logs
    const logsContainer = document.getElementById('training-logs');
    if (logsContainer) {
        logsContainer.innerHTML = '<p>Initializing training...</p>';
    }
    
    // Reset chart data
    if (trainingChart) {
        trainingChart.data.labels = [];
        trainingChart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        trainingChart.update();
    }
    
    // Send training request
    fetch('/api/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            dataset_path: datasetPath,
            epochs: epochs,
            batch_size: batchSize,
            img_size: imgSize,
            model_type: modelType,
            pretrained: pretrained
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store training ID and start progress monitoring
            trainingId = data.training_id;
            hideLoading();
            updateTrainingControls('starting');
            startProgressMonitoring();
        } else {
            showAlert('Failed to start training: ' + data.error, 'danger');
            hideLoading();
            document.getElementById('training-interface').classList.remove('d-none');
            document.getElementById('training-progress').classList.add('d-none');
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        showAlert('Error starting training: ' + error.message, 'danger');
        hideLoading();
        document.getElementById('training-interface').classList.remove('d-none');
        document.getElementById('training-progress').classList.add('d-none');
    });
}

/**
 * Start monitoring training progress
 */
function startProgressMonitoring() {
    // Clear any existing interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Check status immediately
    checkTrainingStatus();
    
    // Then check every 5 seconds
    statusCheckInterval = setInterval(checkTrainingStatus, 5000);
}

/**
 * Check the current training status
 */
function checkTrainingStatus() {
    if (!trainingId) return;
    
    fetch(`/api/training_status?training_id=${trainingId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.status.found) {
                updateTrainingProgress(data.status);
            } else {
                console.error('Training status not found:', data.message);
                if (statusCheckInterval) {
                    clearInterval(statusCheckInterval);
                }
            }
        })
        .catch(error => {
            console.error('Error checking training status:', error);
        });
}

/**
 * Update the training progress UI
 */
function updateTrainingProgress(status) {
    trainingStatus = status;
    
    // Update progress bar
    const progressBar = document.getElementById('training-progress-bar');
    if (progressBar) {
        progressBar.style.width = `${status.progress}%`;
        progressBar.setAttribute('aria-valuenow', status.progress);
        progressBar.textContent = `${Math.round(status.progress)}%`;
        
        // Change color based on status
        progressBar.className = 'progress-bar';
        if (status.status === 'completed') {
            progressBar.classList.add('bg-success');
        } else if (status.status === 'failed') {
            progressBar.classList.add('bg-danger');
        } else {
            progressBar.classList.add('bg-primary');
        }
    }
    
    // Update status text
    const statusText = document.getElementById('training-status-text');
    if (statusText) {
        let text = '';
        switch (status.status) {
            case 'starting':
                text = 'Starting training...';
                break;
            case 'preparing':
                text = 'Preparing training environment...';
                break;
            case 'training':
                text = `Training in progress - Epoch ${status.current_epoch}/${status.total_epochs}`;
                break;
            case 'completed':
                text = 'Training completed successfully';
                break;
            case 'failed':
                text = 'Training failed: ' + (status.error || 'Unknown error');
                break;
            default:
                text = status.status;
        }
        
        statusText.textContent = text;
    }
    
    // Update training controls
    updateTrainingControls(status.status);
    
    // Update logs
    updateTrainingLogs(status.log || []);
    
    // Update metrics chart if training is in progress or completed
    if ((status.status === 'training' || status.status === 'completed') && status.metrics) {
        updateMetricsChart(status);
    }
    
    // If training is complete or failed, stop checking
    if (status.status === 'completed' || status.status === 'failed') {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
        
        // If completed, show model information
        if (status.status === 'completed' && status.model_path) {
            displayModelInfo(status.model_path);
        }
    }
}

/**
 * Update the training logs display
 */
function updateTrainingLogs(logs) {
    const logsContainer = document.getElementById('training-logs');
    if (!logsContainer) return;
    
    // Clear and re-populate logs
    logsContainer.innerHTML = '';
    
    logs.forEach(log => {
        const logLine = document.createElement('p');
        logLine.textContent = log;
        logsContainer.appendChild(logLine);
    });
    
    // Scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

/**
 * Update the metrics chart with new data
 */
function updateMetricsChart(status) {
    if (!trainingChart || !status.metrics) return;
    
    // Extract epoch number
    const epoch = status.current_epoch;
    
    // Update labels if needed
    if (!trainingChart.data.labels.includes(epoch)) {
        trainingChart.data.labels.push(epoch);
    }
    
    // Update datasets
    const metrics = status.metrics;
    
    // Map the metrics to the datasets
    const metricMap = {
        'metrics/mAP50(B)': 0,
        'metrics/mAP50-95(B)': 1,
        'metrics/precision(B)': 2,
        'metrics/recall(B)': 3
    };
    
    // Update each dataset
    Object.keys(metricMap).forEach(metricKey => {
        if (metrics[metricKey] !== undefined) {
            const datasetIndex = metricMap[metricKey];
            // Find the array index for this epoch
            const labelIndex = trainingChart.data.labels.indexOf(epoch);
            
            if (labelIndex >= 0) {
                // Update existing value
                trainingChart.data.datasets[datasetIndex].data[labelIndex] = metrics[metricKey];
            } else {
                // Add new value
                trainingChart.data.datasets[datasetIndex].data.push(metrics[metricKey]);
            }
        }
    });
    
    // Update the chart
    trainingChart.update();
}

/**
 * Update the training control buttons based on status
 */
function updateTrainingControls(status) {
    const exportBtn = document.getElementById('export-model-btn');
    const restartBtn = document.getElementById('restart-training-btn');
    
    if (exportBtn && restartBtn) {
        switch (status) {
            case 'completed':
                exportBtn.disabled = false;
                restartBtn.disabled = false;
                break;
            case 'failed':
                exportBtn.disabled = true;
                restartBtn.disabled = false;
                break;
            default:
                exportBtn.disabled = true;
                restartBtn.disabled = true;
        }
    }
}

/**
 * Display information about the trained model
 */
function displayModelInfo(modelPath) {
    const modelInfoContainer = document.getElementById('trained-model-info');
    if (!modelInfoContainer) return;
    
    // Show model information card
    modelInfoContainer.classList.remove('d-none');
    
    // Update model details
    const modelNameElem = document.getElementById('trained-model-name');
    if (modelNameElem) {
        const modelName = modelPath.split('/').pop();
        modelNameElem.textContent = modelName;
    }
    
    // Update model path
    const modelPathElem = document.getElementById('trained-model-path');
    if (modelPathElem) {
        modelPathElem.textContent = modelPath;
    }
    
    // Store model path for export
    document.getElementById('export-model-btn').setAttribute('data-model-path', modelPath);
}

/**
 * Export the trained model to different formats
 */
function exportModel() {
    const modelPath = document.getElementById('export-model-btn').getAttribute('data-model-path');
    if (!modelPath) {
        showAlert('No model available for export', 'warning');
        return;
    }
    
    // Get selected export format
    const formatSelect = document.getElementById('export-format');
    const format = formatSelect ? formatSelect.value : 'torchscript';
    
    showLoading(`Exporting model to ${format} format...`);
    
    fetch('/api/export_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_path: modelPath,
            format: format
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(`Model exported successfully to ${format} format: ${data.export_path}`, 'success');
        } else {
            showAlert('Failed to export model: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error exporting model:', error);
        showAlert('Error exporting model: ' + error.message, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

/**
 * Restart the training interface
 */
function restartTraining() {
    // Reset training ID and interval
    trainingId = null;
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // Show training interface, hide progress
    document.getElementById('training-interface').classList.remove('d-none');
    document.getElementById('training-progress').classList.add('d-none');
    
    // Hide model info
    const modelInfoContainer = document.getElementById('trained-model-info');
    if (modelInfoContainer) {
        modelInfoContainer.classList.add('d-none');
    }
    
    // Reset chart
    if (trainingChart) {
        trainingChart.data.labels = [];
        trainingChart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        trainingChart.update();
    }
}
