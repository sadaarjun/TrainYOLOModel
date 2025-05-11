# Raspberry Pi Deployment Guide

This document provides detailed instructions for deploying your trained license plate detection models on Raspberry Pi devices.

## Required Dependencies

### 1. Base System Requirements

```bash
# Update your system
sudo apt update
sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-opencv cmake build-essential git
```

### 2. For YOLOv11 & Ultralytics

```bash
# Install latest ultralytics for YOLOv11 support
pip3 install ultralytics>=8.1.0

# Install optional dependencies for better performance
pip3 install onnx onnxruntime
```

### 3. For NCNN Format (Recommended for Best Performance)

```bash
# Install ncnn dependencies
sudo apt install -y libopencv-dev libprotobuf-dev protobuf-compiler

# Clone and build NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig

# Install Python bindings
pip3 install ncnn
```

## Model Format Comparison

| Format | Speed | Memory Usage | Compatibility | Recommended for |
|--------|-------|--------------|--------------|-----------------|
| NCNN   | ✅ Fast | ✅ Low       | ⚠️ Requires setup | Raspberry Pi |
| ONNX   | ⚠️ Medium | ⚠️ Medium   | ✅ Good | General use |
| PyTorch (.pt) | ❌ Slow | ❌ High | ✅ Good | Development |

## Using NCNN Models on Raspberry Pi

### Method 1: Python (with ncnn)

```python
import cv2
import ncnn
import numpy as np

# Load model
net = ncnn.Net()
net.load_param("your_model.param")  # The .param file
net.load_model("your_model.bin")    # The .bin file

# Load and preprocess image
img = cv2.imread("test_image.jpg")
h, w = img.shape[:2]
mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.BGR, w, h, 640, 640)

# Normalize
mean_vals = [0, 0, 0]
norm_vals = [1/255.0, 1/255.0, 1/255.0]
mat_in.substract_mean_normalize(mean_vals, norm_vals)

# Create extractor
ex = net.create_extractor()
ex.input("images", mat_in)

# Run inference
ret, mat_out = ex.extract("output")

# Process detections
for i in range(mat_out.h):
    values = mat_out.row(i)
    class_id, confidence = int(values[0]), values[1]
    x1, y1, x2, y2 = values[2:6]
    
    # Scale coordinates back to original image
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    
    # Draw bounding box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Put text
    label = f"Plate: {confidence:.2f}"
    cv2.putText(img, label, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show result
cv2.imshow("Result", img)
cv2.waitKey(0)
```

### Method 2: C++ (Advanced Performance)

For even better performance, consider using the C++ API directly. A sample application can be found in the NCNN examples directory.

## Troubleshooting

1. **"invalid magic number" error**: The application will automatically repair corrupted model files.

2. **Performance issues**: 
   - Use NCNN format for best performance
   - Lower the resolution (320×320 works well)
   - Use YOLOv11n (nano) or YOLOv8n models
   - Raspberry Pi 4 with 4GB RAM or better is recommended

3. **Missing dependencies**: Follow the installation instructions above

4. **Cannot convert to NCNN format**: Use the online converter at https://convertmodel.com/ as an alternative

## Additional Resources

- NCNN documentation: https://github.com/Tencent/ncnn/wiki
- YOLOv11 paper: https://arxiv.org/abs/2402.10376
- License plate detection datasets: https://universe.roboflow.com/search?q=license%20plate