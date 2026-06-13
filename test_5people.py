#!/usr/bin/env python3
"""
LFW 5-Person Accuracy Test - With Image Preprocessing
Fixes detection issues by upscaling and enhancing images
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

class DirectFaceRecognizer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        det_path = self.model_path / "det_10g.onnx"
        rec_path = self.model_path / "w600k_r50.onnx"
        
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")
        if not rec_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_path}")
        
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        self.det_input = self.det_session.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.det_session.get_outputs()]
        self.rec_input = self.rec_session.get_inputs()[0].name
        self.rec_output = self.rec_session.get_outputs()[0].name
        
        print(f"✓ Models loaded")
    
    def preprocess_image(self, image):
        """Enhance small grayscale images for face detection"""
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Upscale small images (LFW images are 62x47 - too small)
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            # Upscale by 3x for better detection
            new_w, new_h = w * 3, h * 3
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def get_embedding(self, image):
        """Extract face embedding from preprocessed image"""
        try:
            # Preprocess the image first
            processed = self.preprocess_image(image)
            h, w = processed.shape[:2]
            
            # Resize for detection
            img_resized = cv2.resize(processed, (640, 640))
            img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
            img_resized = (img_resized - 127.5) / 128.0
            img_resized = np.expand_dims(img_resized, axis=0)
            
            # Run detection
            detections = self.det_session.run(self.det_outputs, {self.det_input: img_resized})
            
            boxes = detections[0]
            if len(boxes.shape) == 4:
                boxes = boxes[0]
            
            # Find best face
            best_box = None
            best_score = 0
            
            for i in range(boxes.shape[0]):
                if boxes.shape[1] >= 4:
                    x1 = float(boxes[i][0]) * w / 640
                    y1 = float(boxes[i][1]) * h / 640
                    x2 = float(boxes[i][2]) * w / 640
                    y2 = float(boxes[i][3]) * h / 640
                    score = float(boxes[i][4]) if boxes.shape[1] >= 5 else 0.5
                    
                    # Lower threshold for small faces
                    if score > 0.3 and score > best_score:
                        best_score = score
                        best_box = (int(x1), int(y1), int(x2), int(y2))
            
            if best_box:
                x1, y1, x2, y2 = best_box
                # Add padding
                padding = int((x2 - x1) * 0.2)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                face = processed[y1:y2, x1:x2]
                if face.size > 0:
                    # Get embedding
                    face_resized = cv2.resize(face, (112, 112))
                    face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
                    face_resized = (face_resized - 127.5) / 128.0
                    face_resized = np.expand_dims(face_resized, axis=0)
                    
                    embedding = self.rec_session.run([self.rec_output], {self.rec_input: face_resized})[0]
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    return embedding.flatten()
            
            return None
            
        except Exception as e:
            return None
    
    def compare(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return -1
        return float(np.dot(emb1, emb2))

def test_image(person_name, img, recognizer, idx):
    """Test a single image and return if face detected"""
    emb = recognizer.get_embedding(img)
    if emb is not None:
        print(f"  ✓ {person_name}: image {idx+1} detected")
    else:
        print(f"  ✗ {person_name}: image {idx+1} NOT detected")
    return emb

print("="*60)
print("LFW 5-PERSON ACCURACY TEST (WITH PREPROCESSING)")
print("="*60)

# Load model
print("\n[1/4] Loading model...")
model_path = Path(__file__).parent / "models" / "buffalo_l"
recognizer = DirectFaceRecognizer(model_path)

# Load dataset
print("\n[2/4] Loading LFW 5-person subset...")
dataset_path = Path(__file__).parent / "datasets" / "lfw_5people"

if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    sys.exit(1)

# Load images
persons = {}
for person_dir in dataset_path.iterdir():
    if person_dir.is_dir():
        images = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        persons[person_dir.name] = images
        print(f"  ✓ {person_dir.name}: {len(images)} images")

print(f"\n✓ Loaded {len(persons)} people")

# First, test face detection on first few images
print("\n[3/4] Testing face detection on first 3 images...")
print("(This helps verify images are being processed correctly)")

detection_stats = {}
for person_name, images in persons.items():
    detected = 0
    for i in range(min(3, len(images))):
        emb = recognizer.get_embedding(images[i])
        if emb is not None:
            detected += 1
    detection_stats[person_name] = detected
    print(f"  {person_name}: {detected}/3 faces detected")

# Check if we have enough detections
total_detected = sum(detection_stats.values())
if total_detected < 5:
    print("\n⚠️ Low face detection rate. Trying with different parameters...")
    print("The LFW images may be too challenging for the model.")
    print("\nAlternative: Use the Olivetti dataset which is already aligned.")
    sys.exit(1)

# Build gallery (first 5 images that have faces)
print("\n[4/4] Building gallery and running test...")

gallery = {}
test_results = []

for person_name, images in persons.items():
    # Find first 5 images with faces for gallery
    gallery_embs = []
    gallery_indices = []
    
    for i, img in enumerate(images):
        if len(gallery_embs) >= 5:
            break
        emb = recognizer.get_embedding(img)
        if emb is not None:
            gallery_embs.append(emb)
            gallery_indices.append(i)
    
    if gallery_embs:
        gallery[person_name] = np.mean(gallery_embs, axis=0)
        print(f"  ✓ {person_name}: gallery built with {len(gallery_embs)} images")
        
        # Test on images not used in gallery
        for i, img in enumerate(images):
            if i in gallery_indices:
                continue
            test_emb = recognizer.get_embedding(img)
            if test_emb is not None:
                # Find best match
                best_match = None
                best_score = -1
                for gname, gemb in gallery.items():
                    score = recognizer.compare(test_emb, gemb)
                    if score > best_score:
                        best_score = score
                        best_match = gname
                
                test_results.append({
                    'true': person_name,
                    'pred': best_match,
                    'score': best_score
                })
    else:
        print(f"  ✗ {person_name}: No faces detected, skipping")

# Calculate accuracy
if test_results:
    correct = sum(1 for r in test_results if r['true'] == r['pred'])
    total = len(test_results)
    accuracy = (correct / total * 100)
else:
    correct = 0
    total = 0
    accuracy = 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total tests:        {total}")
print(f"Correct:            {correct}")
print(f"Failed:             {total - correct}")
print(f"\n✅ ACCURACY: {accuracy:.1f}%")

if accuracy > 95:
    print("\n✓ CLAIM VALIDATED: >95% accuracy")
elif accuracy > 80 and total > 0:
    print(f"\n⚠ Partial success: {accuracy:.1f}% - {total} tests completed")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "lfw_preprocessed.txt", "w") as f:
    f.write("LFW 5-Person Test (With Preprocessing)\n")
    f.write("=====================================\n\n")
    f.write(f"People tested: {list(persons.keys())}\n")
    f.write(f"Face detection rate: {sum(detection_stats.values())}/15\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")

print(f"\n✓ Report saved: test_results/lfw_preprocessed.txt")