import os
import logging
import uuid
import random
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import model
import utils
import shutil
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "yolo_model_training_secret")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['DATASET_FOLDER'] = 'datasets'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER'], app.config['DATASET_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    
# Subdirectories for datasets
for subfolder in ['train', 'val']:
    for content in ['images', 'labels']:
        os.makedirs(os.path.join(app.config['DATASET_FOLDER'], subfolder, content), exist_ok=True)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/train')
def train():
    """Render the training page."""
    models = utils.get_available_models()
    datasets = utils.get_available_datasets()
    return render_template('train.html', models=models, datasets=datasets)

@app.route('/test')
def test():
    """Render the testing page."""
    models = utils.get_available_models()
    return render_template('test.html', models=models)

@app.route('/settings')
def settings():
    """Render the settings page."""
    return render_template('settings.html')
    
@app.route('/install')
def install():
    """Render the installation guide page."""
    return render_template('install.html')
    
@app.route('/datasets')
def datasets():
    """Render the dataset creation page."""
    datasets = utils.get_available_datasets()
    models = utils.get_available_models()
    return render_template('datasets.html', datasets=datasets, models=models)

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """Handle model uploads."""
    if 'model_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and utils.allowed_model_file(file.filename):
        filename = secure_filename(file.filename)
        model_path = os.path.join(app.config['MODELS_FOLDER'], filename)
        file.save(model_path)
        return jsonify({'success': True, 'model_path': model_path})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Handle test image uploads."""
    if 'image_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and utils.allowed_image_file(file.filename):
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        return jsonify({'success': True, 'image_path': image_path})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset uploads (zip file)."""
    if 'dataset_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['dataset_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and utils.allowed_dataset_file(file.filename):
        dataset_id = str(uuid.uuid4())
        dataset_dir = os.path.join(app.config['DATASET_FOLDER'], dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}.zip")
        file.save(zip_path)
        
        try:
            utils.extract_and_organize_dataset(zip_path, dataset_dir)
            os.remove(zip_path)  # Clean up the zip file
            return jsonify({'success': True, 'dataset_id': dataset_id})
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            shutil.rmtree(dataset_dir, ignore_errors=True)
            return jsonify({'success': False, 'error': f'Error processing dataset: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/api/test_model', methods=['POST'])
def test_model():
    """Test a model on an image."""
    data = request.json
    model_path = data.get('model_path')
    image_path = data.get('image_path')
    confidence = float(data.get('confidence', 0.25))
    
    if not model_path or not image_path:
        return jsonify({'success': False, 'error': 'Missing model or image path'})
    
    try:
        # Run inference with the model
        results, output_image_path = model.run_inference(model_path, image_path, confidence)
        
        # Return the detection results and the path to the annotated image
        return jsonify({
            'success': True, 
            'detections': results, 
            'output_image': output_image_path
        })
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return jsonify({'success': False, 'error': f'Error during inference: {str(e)}'})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start training a YOLO model."""
    data = request.json
    dataset_path = data.get('dataset_path')
    epochs = int(data.get('epochs', 100))
    batch_size = int(data.get('batch_size', 16))
    img_size = int(data.get('img_size', 640))
    model_type = data.get('model_type', 'yolov8n')
    pretrained = data.get('pretrained', True)
    
    if not dataset_path:
        return jsonify({'success': False, 'error': 'Missing dataset path'})
    
    try:
        # Start training in a separate thread
        training_id = str(uuid.uuid4())
        session['training_id'] = training_id
        
        model.start_training(
            dataset_path, 
            epochs, 
            batch_size, 
            img_size, 
            model_type, 
            pretrained,
            training_id
        )
        
        return jsonify({
            'success': True,
            'training_id': training_id,
            'message': 'Training started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'success': False, 'error': f'Error starting training: {str(e)}'})

@app.route('/api/training_status', methods=['GET'])
def training_status():
    """Get the status of the current training job."""
    training_id = request.args.get('training_id') or session.get('training_id')
    if not training_id:
        return jsonify({'success': False, 'error': 'No training job found'})
    
    status = model.get_training_status(training_id)
    return jsonify({'success': True, 'status': status})

@app.route('/api/export_model', methods=['POST'])
def export_model():
    """Export a trained model to various formats."""
    data = request.json
    model_path = data.get('model_path')
    export_format = data.get('format', 'torchscript')  # Options: torchscript, onnx, openvino, tflite
    
    if not model_path:
        return jsonify({'success': False, 'error': 'Missing model path'})
    
    try:
        export_path = model.export_model(model_path, export_format)
        return jsonify({
            'success': True,
            'export_path': export_path,
            'message': f'Model exported to {export_format} format successfully'
        })
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        return jsonify({'success': False, 'error': f'Error exporting model: {str(e)}'})

@app.route('/api/get_system_info', methods=['GET'])
def get_system_info():
    """Get system information (CPU, RAM, etc.)."""
    try:
        system_info = utils.get_system_info()
        return jsonify({'success': True, 'system_info': system_info})
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({'success': False, 'error': f'Error getting system info: {str(e)}'})

@app.route('/api/create_dataset', methods=['POST'])
def create_dataset():
    """Create a new dataset structure."""
    dataset_name = request.json.get('dataset_name', 'custom_dataset')
    
    try:
        dataset_path = utils.create_empty_dataset(dataset_name, app.config['DATASET_FOLDER'])
        return jsonify({
            'success': True, 
            'dataset_path': dataset_path,
            'message': 'Empty dataset structure created successfully'
        })
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error creating dataset: {str(e)}'})

@app.route('/api/auto_annotate', methods=['POST'])
def auto_annotate():
    """Auto-annotate images for license plates using the selected model."""
    if 'images_folder' not in request.files and 'image_file' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'})
    
    data = request.form
    model_path = data.get('model_path')
    confidence = float(data.get('confidence', 0.25))
    dataset_name = data.get('dataset_name', f'auto_dataset_{int(time.time())}')
    
    # Create the dataset directory
    try:
        dataset_path = utils.create_empty_dataset(dataset_name, app.config['DATASET_FOLDER'])
        images_dir = os.path.join(dataset_path, 'images')
        labels_dir = os.path.join(dataset_path, 'labels')
        train_dir = os.path.join(images_dir, 'train')
        val_dir = os.path.join(images_dir, 'val')
        test_dir = os.path.join(images_dir, 'test')
        
        # Process a single image
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No selected file'})
            
            if file and utils.allowed_image_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(train_dir, filename)
                file.save(image_path)
                
                # Run detection on the image
                success, annotations = utils.annotate_image(image_path, model_path, confidence)
                if success:
                    # Save annotation as YOLO format
                    utils.save_yolo_annotation(image_path, annotations, labels_dir)
                
                return jsonify({
                    'success': True,
                    'dataset_path': dataset_path,
                    'images_processed': 1,
                    'message': 'Image annotated and dataset created successfully'
                })
        
        # Process a folder of images
        elif 'images_folder' in request.files:
            files = request.files.getlist('images_folder')
            processed_count = 0
            
            for file in files:
                if file.filename == '':
                    continue
                
                if utils.allowed_image_file(file.filename):
                    filename = secure_filename(file.filename)
                    # Distribute images between train/val/test (80/10/10)
                    rand_val = random.random()
                    if rand_val < 0.8:
                        image_path = os.path.join(train_dir, filename)
                    elif rand_val < 0.9:
                        image_path = os.path.join(val_dir, filename)
                    else:
                        image_path = os.path.join(test_dir, filename)
                        
                    file.save(image_path)
                    
                    # Run detection on the image
                    success, annotations = utils.annotate_image(image_path, model_path, confidence)
                    if success:
                        # Save annotation as YOLO format
                        utils.save_yolo_annotation(image_path, annotations, labels_dir)
                        processed_count += 1
            
            return jsonify({
                'success': True,
                'dataset_path': dataset_path,
                'images_processed': processed_count,
                'message': f'{processed_count} images annotated and dataset created successfully'
            })
        
        return jsonify({'success': False, 'error': 'No valid files found'})
        
    except Exception as e:
        logger.error(f"Error auto-annotating images: {str(e)}")
        return jsonify({'success': False, 'error': f'Error auto-annotating images: {str(e)}'})
        
@app.route('/api/manual_annotate', methods=['POST'])
def manual_annotate():
    """Save manual annotations to create a dataset."""
    data = request.json
    image_path = data.get('image_path')
    annotations = data.get('annotations', [])
    dataset_name = data.get('dataset_name', f'manual_dataset_{int(time.time())}')
    
    try:
        # Create dataset if it doesn't exist
        dataset_path = utils.create_empty_dataset(dataset_name, app.config['DATASET_FOLDER'])
        images_dir = os.path.join(dataset_path, 'images', 'train')
        labels_dir = os.path.join(dataset_path, 'labels', 'train')
        
        # Copy image to dataset
        if image_path and os.path.exists(image_path):
            filename = os.path.basename(image_path)
            dest_path = os.path.join(images_dir, filename)
            shutil.copy(image_path, dest_path)
            
            # Save annotations
            if annotations:
                utils.save_yolo_annotation(dest_path, annotations, labels_dir)
            
            return jsonify({
                'success': True,
                'dataset_path': dataset_path,
                'message': 'Image annotated and saved to dataset successfully'
            })
        
        return jsonify({'success': False, 'error': 'Image path not valid'})
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error creating dataset: {str(e)}'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
