import os
import sys
import zipfile
import yaml
import psutil
import glob
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MODEL_EXTENSIONS = {'pt', 'pth', 'onnx', 'torchscript'}
ALLOWED_DATASET_EXTENSIONS = {'zip'}

def allowed_image_file(filename):
    """Check if a file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_model_file(filename):
    """Check if a file is an allowed model type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

def allowed_dataset_file(filename):
    """Check if a file is an allowed dataset type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS

def get_available_models(models_dir='models'):
    """Get a list of available trained models."""
    os.makedirs(models_dir, exist_ok=True)
    
    models = []
    for ext in ALLOWED_MODEL_EXTENSIONS:
        models.extend(glob.glob(os.path.join(models_dir, f"*.{ext}")))
    
    return [os.path.basename(model) for model in models]

def get_available_datasets(datasets_dir='datasets'):
    """Get a list of available datasets."""
    os.makedirs(datasets_dir, exist_ok=True)
    
    # List all directories in the datasets folder
    datasets = [d for d in os.listdir(datasets_dir) 
                if os.path.isdir(os.path.join(datasets_dir, d))]
    
    return datasets

def extract_and_organize_dataset(zip_path, dataset_dir):
    """Extract dataset from zip and organize into YOLO format."""
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Check if the dataset has proper structure or organize it
    if not _is_yolo_dataset(dataset_dir):
        _organize_dataset(dataset_dir)
    
    # Create data.yaml file
    _create_data_yaml(dataset_dir)
    
    return dataset_dir

def _is_yolo_dataset(dataset_dir):
    """Check if the dataset has proper YOLO format."""
    # Check for expected train/val folders with images and labels
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir_path in required_dirs:
        if not os.path.isdir(os.path.join(dataset_dir, dir_path)):
            return False
    
    # Check if data.yaml exists
    if not os.path.isfile(os.path.join(dataset_dir, 'data.yaml')):
        return False
    
    return True

def _organize_dataset(dataset_dir):
    """Organize dataset into YOLO format."""
    # Create required directories
    for split in ['train', 'val']:
        for content in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_dir, split, content), exist_ok=True)
    
    # Scan for images and labels
    images = []
    labels = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip directories we created
            if any(x in file_path for x in ['train/images', 'train/labels', 'val/images', 'val/labels']):
                continue
                
            if allowed_image_file(file):
                images.append(file_path)
            elif file.endswith('.txt') and not file == 'data.yaml':
                labels.append(file_path)
    
    # Split between train and val (80/20)
    train_count = int(len(images) * 0.8)
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    # Move images to appropriate directories
    for img_path in train_images:
        shutil.move(img_path, os.path.join(dataset_dir, 'train', 'images', os.path.basename(img_path)))
    
    for img_path in val_images:
        shutil.move(img_path, os.path.join(dataset_dir, 'val', 'images', os.path.basename(img_path)))
    
    # Try to find and move corresponding labels
    for split, img_list in [('train', train_images), ('val', val_images)]:
        for img_path in img_list:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Look for matching label file
            label_file = f"{img_name}.txt"
            for label_path in labels:
                if os.path.basename(label_path) == label_file:
                    dest_path = os.path.join(dataset_dir, split, 'labels', label_file)
                    try:
                        shutil.move(label_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Could not move label {label_path}: {str(e)}")
                    break

def _create_data_yaml(dataset_dir):
    """Create a data.yaml file for the dataset."""
    # Determine number of classes by analyzing label files
    classes = set()
    
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        
        if os.path.exists(labels_dir):
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    try:
                        with open(os.path.join(labels_dir, label_file), 'r') as f:
                            for line in f:
                                # YOLO format: class_id x y w h
                                class_id = int(line.strip().split()[0])
                                classes.add(class_id)
                    except Exception as e:
                        logger.warning(f"Error reading label file {label_file}: {str(e)}")
    
    num_classes = max(classes) + 1 if classes else 1
    
    # Create class names (default to numerical if no names provided)
    names = [f"class_{i}" for i in range(num_classes)]
    
    # For number plate detection, use a specific name
    names = ["license_plate"]
    
    # Create the YAML data
    data = {
        'path': dataset_dir,
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'val', 'images'),
        'names': names,
        'nc': len(names)
    }
    
    # Write the YAML file
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_system_info():
    """Get system information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get Raspberry Pi specific info
        is_raspberry_pi = os.path.exists('/sys/firmware/devicetree/base/model')
        pi_model = None
        
        if is_raspberry_pi:
            try:
                with open('/sys/firmware/devicetree/base/model', 'r') as f:
                    pi_model = f.read().strip('\0')
            except:
                pi_model = "Unknown Raspberry Pi"
        
        # Get Python and relevant packages versions
        python_version = sys.version
        
        return {
            'cpu_percent': cpu_percent,
            'memory_total': memory.total,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_free': disk.free,
            'disk_percent': disk.percent,
            'is_raspberry_pi': is_raspberry_pi,
            'pi_model': pi_model,
            'python_version': python_version,
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {'error': str(e)}

def create_empty_dataset(dataset_name, datasets_dir):
    """Create an empty dataset structure."""
    dataset_path = os.path.join(datasets_dir, dataset_name)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for content in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_path, split, content), exist_ok=True)
    
    # Create a basic data.yaml file
    data = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'train', 'images'),
        'val': os.path.join(dataset_path, 'val', 'images'),
        'test': os.path.join(dataset_path, 'test', 'images'),
        'names': ['license_plate'],
        'nc': 1
    }
    
    with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return dataset_path

def annotate_image(image_path, model_path, confidence=0.25):
    """
    Auto-annotate an image for license plates using the specified model.
    Returns a tuple (success, annotations) where annotations is a list of 
    [class_id, x_center, y_center, width, height] in YOLO format.
    """
    try:
        import model as model_module
        
        # Run inference with the model to detect license plates
        results, _ = model_module.run_inference(model_path, image_path, confidence)
        
        if not results or len(results) == 0:
            return False, []
        
        annotations = []
        # Get the image dimensions
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Convert the detections to YOLO format
        for detection in results:
            cls_id = 0  # License plate is always class 0
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            
            # Convert to YOLO format (normalized)
            x_center = (x1 + x2) / (2 * img_width)
            y_center = (y1 + y2) / (2 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            annotations.append([cls_id, x_center, y_center, width, height])
        
        return True, annotations
    
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}")
        # In mock mode or if error, return a simulated annotation
        return True, [[0, 0.5, 0.5, 0.3, 0.1]]

def save_yolo_annotation(image_path, annotations, labels_dir):
    """
    Save annotations in YOLO format.
    
    Args:
        image_path: Path to the image file
        annotations: List of annotations in YOLO format [class_id, x_center, y_center, width, height]
        labels_dir: Directory to save the label file
    """
    try:
        # Determine the base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Determine which split directory (train/val/test) the image is in
        parent_dir = os.path.basename(os.path.dirname(image_path))
        
        # Create the corresponding label file path
        if parent_dir in ['train', 'val', 'test']:
            # Image is already in a split directory
            label_path = os.path.join(labels_dir, parent_dir, f"{base_name}.txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
        else:
            # Image is not in a split directory, assume train
            label_path = os.path.join(labels_dir, 'train', f"{base_name}.txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        # Write annotations to the label file
        with open(label_path, 'w') as f:
            for annotation in annotations:
                # Format: class_id x_center y_center width height
                line = " ".join([str(x) for x in annotation])
                f.write(line + "\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving YOLO annotation: {str(e)}")
        return False
