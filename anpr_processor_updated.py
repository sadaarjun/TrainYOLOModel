import cv2
import numpy as np
import logging
import re
import pytesseract
import os
import sys
from pathlib import Path

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anpr_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check if we can import YOLO
try:
    from ultralytics import YOLO
    MOCK_MODE = False
    logger.info("Successfully imported ultralytics YOLO")
except ImportError:
    logger.warning("Unable to import ultralytics YOLO, using fallback mechanism")
    MOCK_MODE = True

class ANPRSettings:
    """Class to hold ANPR configuration settings"""
    def __init__(self):
        self.enable_preprocessing = True
        self.min_plate_size = 20  # Minimum plate area (pixels)
        self.max_plate_size = 100000  # Maximum plate area (pixels)
        self.yolo_confidence = 0.5  # YOLOv11 detection confidence threshold
        self.model_path = "numberplate-best.pt"  # Path to YOLOv11 model
        self.use_ncnn = False  # Whether to use NCNN format (for Raspberry Pi)
        self.ncnn_param_path = None  # Path to NCNN param file
        self.ncnn_bin_path = None  # Path to NCNN bin file

def process_anpr(image, anpr_settings):
    """
    Process an image for HSRP license plate detection and recognition using YOLOv11
    """
    try:
        logger.debug("Starting ANPR processing")
        if image is None or image.size == 0:
            logger.error("Invalid image input: Image is None or empty")
            return False, "Invalid image input"

        height, width = image.shape[:2]
        logger.debug(f"Input image dimensions: {width}x{height}")

        max_width = 1000  # Increased for better resolution
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            image = cv2.resize(image, (max_width, new_height))
            logger.debug(f"Resized image to: {max_width}x{new_height}")
        else:
            logger.debug("No resizing needed")

        img = image.copy()

        if anpr_settings.enable_preprocessing:
            logger.debug("Applying preprocessing")
            img = preprocess_image(img)
        else:
            logger.debug("Preprocessing disabled")

        logger.debug("Detecting plate region with YOLOv11")
        plate_img = detect_plate_region(img, anpr_settings)

        if plate_img is None:
            logger.warning("No license plate detected in the image")
            return False, "No license plate detected in the image"

        logger.debug("Recognizing plate text")
        plate_text = recognize_plate_text(plate_img)

        if not plate_text:
            logger.warning("Could not recognize text on the license plate")
            return False, "Could not recognize text on the license plate"

        logger.debug(f"Raw plate text: {plate_text}")
        cleaned_plate = clean_plate_text(plate_text)

        if not cleaned_plate:
            logger.warning("Recognized text does not appear to be a valid HSRP license plate")
            return False, "Recognized text does not appear to be a valid HSRP license plate"

        logger.info(f"Successfully detected license plate: {cleaned_plate}")
        return True, cleaned_plate

    except Exception as e:
        logger.error(f"Error in ANPR processing: {str(e)}", exc_info=True)
        return False, f"ANPR Processing Error: {str(e)}"

def preprocess_image(image):
    """
    Apply preprocessing to enhance the image for HSRP license plate detection
    """
    try:
        logger.debug("Starting image preprocessing")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        logger.debug("Converted to HSV for color filtering")

        # Color filtering for HSRP plates (white, yellow, green)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_yellow, mask_green))
        logger.debug("Applied color filtering for white, yellow, green backgrounds")

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted to grayscale after color filtering")

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        logger.debug("Applied Gaussian blur with kernel (5,5)")

        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        logger.debug("Applied adaptive thresholding (inverted, blockSize=11)")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        logger.debug("Applied dilation to strengthen plate edges")

        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "preprocessed.jpg"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "color_mask.jpg"), mask)
        logger.debug("Saved preprocessed and color mask images to debug/")

        return thresh

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}", exc_info=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        logger.debug("Falling back to grayscale image due to preprocessing error")
        return gray

def correct_perspective(image, bbox):
    """
    Apply perspective correction to a YOLOv11-detected bounding box
    """
    try:
        logger.debug("Applying perspective correction")
        x1, y1, x2, y2 = bbox
        
        # Convert to contour-like points for perspective transform
        pts = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        
        # Calculate destination dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Create destination points
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
        
        logger.debug(f"Perspective corrected to {width}x{height}")
        
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "warped_plate.jpg"), warped)
        logger.debug("Saved warped plate to debug/warped_plate.jpg")
        
        return warped
    
    except Exception as e:
        logger.error(f"Error in perspective correction: {str(e)}", exc_info=True)
        # Fallback to simple crop
        return image[int(y1):int(y2), int(x1):int(x2)]

def load_yolov11_model(model_path):
    """
    Load a YOLOv11 model for inference
    """
    if MOCK_MODE:
        logger.warning("Running in mock mode, returning dummy model")
        return None
    
    try:
        # Support for our app's generated models
        # First check if this is a NCNN format model (best for Raspberry Pi)
        if model_path.endswith('.param'):
            try:
                import ncnn
                logger.info(f"Loading NCNN model: {model_path}")
                net = ncnn.Net()
                
                # Load param and bin files
                bin_path = os.path.splitext(model_path)[0] + '.bin'
                if not os.path.exists(bin_path):
                    logger.error(f"NCNN bin file missing: {bin_path}")
                    raise FileNotFoundError(f"NCNN bin file missing: {bin_path}")
                
                # Load the model
                net.load_param(model_path)
                net.load_model(bin_path)
                logger.info(f"Successfully loaded NCNN model: {model_path}")
                return {'type': 'ncnn', 'net': net}
            except ImportError:
                logger.error("Failed to import ncnn package - is it installed?")
                raise
        
        # For PT model files from our app
        # First, check if this is a YOLOv11 model from our application
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        logger.info(f"Successfully loaded YOLO model: {model.model}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading YOLOv11 model: {str(e)}", exc_info=True)
        raise

def detect_plate_region(image, anpr_settings, debug_dir="debug"):
    """
    Detect HSRP license plate region using YOLOv11
    """
    try:
        logger.debug("Starting plate region detection with YOLOv11")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            logger.debug(f"Created debug directory: {debug_dir}")

        height, width = image.shape[:2]
        logger.debug(f"Input image for detection: {width}x{height}")

        # Load model
        try:
            model = load_yolov11_model(anpr_settings.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None

        # Check for NCNN model
        is_ncnn = isinstance(model, dict) and model.get('type') == 'ncnn'
        
        # Convert grayscale/thresholded image back to BGR for YOLOv11
        input_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
        logger.debug("Converted image to BGR for YOLOv11")

        # Perform detection based on model type
        debug_img = input_img.copy()
        plate_img = None
        
        if MOCK_MODE:
            logger.info("Mock mode: Generating simulated detection")
            # Create a simulated detection in the center of the image
            center_x = width // 2
            center_y = height // 2
            w = width // 3
            h = height // 5
            x1, y1 = center_x - w // 2, center_y - h // 2
            x2, y2 = center_x + w // 2, center_y + h // 2
            conf = 0.85
            
            # Create a mock detection
            detection = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': conf, 'class': 0, 'class_name': 'license_plate' 
            }
            
            # Draw bounding box on debug image
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_img, f"Conf: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            # Crop the plate region
            plate_img = input_img[y1:y2, x1:x2]
        
        elif is_ncnn:
            logger.info("Using NCNN model for detection")
            net = model['net']
            
            # Convert image to NCNN format and resize
            img_h, img_w = input_img.shape[:2]
            input_size = 640  # Standard YOLO input size
            
            # Create NCNN mat from image
            mat_in = ncnn.Mat.from_pixels_resize(
                input_img, ncnn.Mat.PixelType.BGR, img_w, img_h, input_size, input_size
            )
            
            # Normalize
            mean_vals = [0, 0, 0]
            norm_vals = [1/255.0, 1/255.0, 1/255.0]
            mat_in.substract_mean_normalize(mean_vals, norm_vals)
            
            # Create extractor
            ex = net.create_extractor()
            ex.input("images", mat_in)  # Standard YOLO input name
            
            # Get output
            ret, mat_out = ex.extract("output")  # Standard YOLO output name
            
            # Process results
            best_detection = None
            
            if ret == 0:
                for i in range(mat_out.h):
                    values = mat_out.row(i)
                    
                    # Skip background detections
                    if len(values) < 6:
                        continue
                    
                    class_id = int(values[0])
                    conf = float(values[1])
                    
                    # Filter by confidence
                    if conf < anpr_settings.yolo_confidence:
                        continue
                    
                    # Get coordinates (normalized)
                    x1 = float(values[2]) * img_w
                    y1 = float(values[3]) * img_h
                    x2 = float(values[4]) * img_w
                    y2 = float(values[5]) * img_h
                    
                    # Calculate area and aspect ratio
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter by size and aspect ratio
                    if (area < anpr_settings.min_plate_size or 
                        area > anpr_settings.max_plate_size or
                        aspect_ratio < 0.5 or aspect_ratio > 10.0):
                        continue
                    
                    # Keep the highest confidence detection
                    if best_detection is None or conf > best_detection['conf']:
                        best_detection = {
                            'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                            'conf': conf, 'class': class_id, 'class_name': 'license_plate'
                        }
                    
            # Process best detection
            if best_detection:
                x1, y1 = best_detection['x1'], best_detection['y1']
                x2, y2 = best_detection['x2'], best_detection['y2']
                conf = best_detection['conf']
                
                # Draw bounding box on debug image
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img, f"Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                
                # Apply perspective correction
                plate_img = correct_perspective(input_img, (x1, y1, x2, y2))
        
        else:
            logger.info("Using YOLO model for detection")
            # Standard YOLO detection
            results = model(input_img, conf=anpr_settings.yolo_confidence, verbose=False)
            logger.debug(f"YOLO detected {len(results[0].boxes)} objects")
            
            # Process detections
            for box in results[0].boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy())
                
                logger.debug(f"Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.2f}, class={cls}")
                
                # Filter by class (assuming class 0 is 'License_Plate' or similar)
                if cls != 0:
                    logger.debug(f"Skipped box (class={cls}, expected 0)")
                    continue
                
                # Calculate width, height and area
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Check plate size
                if area < anpr_settings.min_plate_size or area > anpr_settings.max_plate_size:
                    logger.debug(f"Skipped box (area={area:.0f}, expected {anpr_settings.min_plate_size}-{anpr_settings.max_plate_size})")
                    continue
                
                # Check aspect ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 10.0:
                    logger.debug(f"Skipped box (aspect_ratio={aspect_ratio:.2f}, expected 0.5-10.0)")
                    continue
                
                logger.debug(f"Valid plate detected: area={area:.0f}, aspect_ratio={aspect_ratio:.2f}, conf={conf:.2f}")
                
                # Crop and apply perspective correction
                plate_img = correct_perspective(input_img, (x1, y1, x2, y2))
                
                # Draw bounding box on debug image
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img, f"Conf: {conf:.2f}, AR: {aspect_ratio:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                break  # Take the first valid plate
        
        if debug_img is not None:
            cv2.imwrite(os.path.join(debug_dir, "yolo_detections.jpg"), debug_img)
            logger.debug("Saved YOLOv11 detections to debug/yolo_detections.jpg")
        
        if plate_img is None:
            logger.warning("No valid plate region found after filtering")
        else:
            cv2.imwrite(os.path.join(debug_dir, "detected_plate.jpg"), plate_img)
            logger.debug("Saved detected plate to debug/detected_plate.jpg")
        
        return plate_img

    except Exception as e:
        logger.error(f"Error detecting plate region: {str(e)}", exc_info=True)
        return None

def recognize_plate_text(plate_img):
    """
    Recognize text on HSRP license plate using Tesseract OCR
    """
    try:
        logger.debug("Starting plate text recognition")
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        logger.debug("Applied CLAHE contrast enhancement")

        blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
        logger.debug("Applied bilateral filter for OCR")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        logger.debug("Applied morphological closing")

        binary = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6
        )
        logger.debug("Applied adaptive thresholding for OCR (blockSize=25)")

        binary = cv2.fastNlMeansDenoising(binary, h=15)
        logger.debug("Applied denoising")

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)
        logger.debug("Applied erosion and dilation")

        # Scale up for better OCR
        scale_factor = 2.0
        height, width = binary.shape
        binary = cv2.resize(binary, (int(width * scale_factor), int(height * scale_factor)), 
                            interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Scaled image by {scale_factor}x for OCR")

        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "ocr_input.jpg"), binary)
        logger.debug("Saved OCR input image to debug/ocr_input.jpg")

        # Set Tesseract parameters
        custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Run OCR
        text = pytesseract.image_to_string(binary, config=custom_config).strip()
        logger.debug(f"Raw OCR output: {text}")

        # Also try with negative image
        inverted = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(debug_dir, "ocr_input_inv.jpg"), inverted)
        logger.debug("Saved inverted OCR input to debug/ocr_input_inv.jpg")
        
        text_inv = pytesseract.image_to_string(inverted, config=custom_config).strip()
        logger.debug(f"Raw OCR output (inverted): {text_inv}")

        # Use whichever result is longer
        if len(text_inv) > len(text):
            text = text_inv
            logger.debug("Using inverted image OCR result (longer)")

        # If nothing is recognized, try with different PSM mode
        if not text:
            alt_config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(binary, config=alt_config).strip()
            logger.debug(f"Alternative OCR output (PSM 8): {text}")

        return text

    except Exception as e:
        logger.error(f"Error during text recognition: {str(e)}", exc_info=True)
        return ""

def clean_plate_text(text):
    """
    Clean and validate recognized license plate text for HSRP format
    """
    try:
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        logger.debug(f"Cleaned text: {text}")

        # Check for common OCR errors
        text = text.replace('0', 'O').replace('1', 'I').replace('5', 'S')
        logger.debug(f"After common error correction: {text}")

        # Format for standard Indian license plates: 2 letters + 2 digits + 4 alphanumerics
        plate_pattern = r'^[A-Z]{2}\d{1,2}[A-Z0-9]{1,4}$'
        if re.match(plate_pattern, text):
            logger.debug(f"Valid license plate format: {text}")
            return text
        
        # If no match, try to extract a valid plate from the detected text
        all_plates = re.findall(r'[A-Z]{2}\d{1,2}[A-Z0-9]{1,4}', text)
        if all_plates:
            logger.debug(f"Extracted plate from text: {all_plates[0]}")
            return all_plates[0]
            
        # If still no match, return original text if it's at least 4 characters
        if len(text) >= 4:
            logger.warning(f"Returning unformatted text: {text}")
            return text
            
        logger.warning("No valid license plate text found")
        return ""
        
    except Exception as e:
        logger.error(f"Error cleaning plate text: {str(e)}", exc_info=True)
        # Return raw text as fallback
        return text

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <image_path> <model_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
        
    # Configure settings
    settings = ANPRSettings()
    settings.model_path = model_path
    settings.yolo_confidence = 0.25  # Lower threshold for testing
    
    # Detect and recognize plate
    success, result = process_anpr(image, settings)
    
    if success:
        print(f"Detected license plate: {result}")
    else:
        print(f"Failed to detect license plate: {result}")