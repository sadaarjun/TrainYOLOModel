/**
 * JavaScript file specifically for testing YOLO models
 */

// Initialize testing functionality
document.addEventListener('DOMContentLoaded', function() {
    const testModelBtn = document.getElementById('test-model-btn');
    if (testModelBtn) {
        testModelBtn.addEventListener('click', testModel);
    }
    
    // Setup confidence threshold slider
    const confidenceSlider = document.getElementById('confidence-threshold');
    const confidenceValue = document.getElementById('confidence-value');
    
    if (confidenceSlider && confidenceValue) {
        // Set initial value
        confidenceValue.textContent = confidenceSlider.value;
        
        // Update value when slider changes
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
    }
    
    // Setup model upload form
    const modelUploadForm = document.getElementById('model-upload-form');
    if (modelUploadForm) {
        modelUploadForm.addEventListener('submit', uploadModelFile);
    }
});

/**
 * Upload a model file
 */
function uploadModelFile(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('model-file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        showAlert('Please select a model file to upload', 'warning');
        return;
    }
    
    // Show loading indicator
    showLoading('Uploading model...');
    
    const formData = new FormData();
    formData.append('model_file', fileInput.files[0]);
    
    fetch('/api/upload_model', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Model uploaded successfully', 'success');
            
            // Refresh the page to show the new model in the list
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            showAlert('Failed to upload model: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error uploading model:', error);
        showAlert('Error uploading model: ' + error.message, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}
