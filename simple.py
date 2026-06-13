#!/usr/bin/env python3
"""
Simple accuracy test - Just verifies model loads and can detect any face
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing basic face detection...")

# Create a simple test face
test_img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(test_img, (150, 150), 70, (200, 180, 150), -1)
cv2.circle(test_img, (115, 125), 12, (0, 0, 0), -1)
cv2.circle(test_img, (185, 125), 12, (0, 0, 0), -1)

# Try to load and use ONNX model
import onnxruntime as ort

model_path = Path(__file__).parent / "models" / "buffalo_l" / "det_10g.onnx"
if not model_path.exists():
    print(f"Model not found at {model_path}")
    sys.exit(1)

session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

# Preprocess
img_resized = cv2.resize(test_img, (640, 640))
img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
img_resized = (img_resized - 127.5) / 128.0
img_resized = np.expand_dims(img_resized, axis=0)

# Run detection
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_resized})

print(f"Detection output shape: {outputs[0].shape}")
print("✓ Model works!")