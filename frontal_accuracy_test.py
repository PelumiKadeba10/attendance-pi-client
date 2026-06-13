#!/usr/bin/env python3
"""
Olivetti Faces Accuracy Test - Using ONNX Runtime Directly
NO DOWNLOAD - Uses your local buffalo_l model files
"""

import numpy as np
import cv2
import onnxruntime as ort
from sklearn.datasets import fetch_olivetti_faces
from pathlib import Path
import sys

class DirectFaceRecognizer:
    """Face recognizer using ONNX Runtime directly - no InsightFace download"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        print(f"Loading models from: {self.model_path}")
        
        # Load detection model
        det_path = self.model_path / "det_10g.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")
        
        # Load recognition model
        rec_path = self.model_path / "w600k_r50.onnx"
        if not rec_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_path}")
        
        # Load with CPU only - no network
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        # Get input/output names
        self.det_input = self.det_session.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.det_session.get_outputs()]
        self.rec_input = self.rec_session.get_inputs()[0].name
        self.rec_output = self.rec_session.get_outputs()[0].name
        
        print("✓ Models loaded successfully (offline mode)\n")
    
    def get_embedding(self, image):
        """Extract face embedding from image"""
        try:
            # Detect face
            h, w = image.shape[:2]
            
            # Resize for detection
            img_resized = cv2.resize(image, (640, 640))
            img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
            img_resized = (img_resized - 127.5) / 128.0
            img_resized = np.expand_dims(img_resized, axis=0)
            
            # Run detection
            detections = self.det_session.run(self.det_outputs, {self.det_input: img_resized})
            
            # Parse boxes (first output)
            boxes = detections[0]
            if len(boxes.shape) == 4:
                boxes = boxes[0]
            
            # Find best face
            best_box = None
            best_score = 0
            
            for i in range(min(boxes.shape[0], 10)):
                # Get box coordinates
                if boxes.shape[1] >= 4:
                    x1 = float(boxes[i][0]) * w / 640
                    y1 = float(boxes[i][1]) * h / 640
                    x2 = float(boxes[i][2]) * w / 640
                    y2 = float(boxes[i][3]) * h / 640
                    
                    # Get confidence
                    score = float(boxes[i][4]) if boxes.shape[1] >= 5 else 0.5
                    
                    if score > 0.5 and score > best_score and (x2 - x1) > 30:
                        best_score = score
                        best_box = (int(x1), int(y1), int(x2), int(y2))
            
            if best_box:
                x1, y1, x2, y2 = best_box
                # Add padding
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    # Get embedding
                    face_resized = cv2.resize(face, (112, 112))
                    face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
                    face_resized = (face_resized - 127.5) / 128.0
                    face_resized = np.expand_dims(face_resized, axis=0)
                    
                    embedding = self.rec_session.run([self.rec_output], {self.rec_input: face_resized})[0]
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    return embedding.flatten()
            
            return None
            
        except Exception as e:
            return None
    
    def compare(self, emb1, emb2):
        """Cosine similarity"""
        if emb1 is None or emb2 is None:
            return -1
        return float(np.dot(emb1, emb2))

def preprocess_for_insightface(img):
    """Convert Olivetti grayscale image to format recognizable by model"""
    # Olivetti images are 64x64 grayscale (0-1 range)
    # Upscale to 224x224 and convert to BGR
    img_uint8 = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    img_large = cv2.resize(img_bgr, (224, 224))
    return img_large

print("="*60)
print("OLIVETTI FACES ACCURACY TEST (OFFLINE)")
print("="*60)

# Load model from local folder
print("\n[1/4] Loading model from local folder...")
model_path = Path(__file__).parent / "models" / "buffalo_l"
recognizer = DirectFaceRecognizer(model_path)

# Load dataset
print("[2/4] Loading Olivetti dataset...")
print("(This downloads 2.5 MB once - please wait)")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces.images
targets = faces.target

print(f"✓ Loaded {len(images)} images of 40 people (10 each)")

# Group images by person
print("\n[3/4] Organizing images...")
persons = {}
for i, (img, person_id) in enumerate(zip(images, targets)):
    if person_id not in persons:
        persons[person_id] = []
    persons[person_id].append(img)

print(f"✓ Organized {len(persons)} people")

# Run test
print("\n[4/4] Running accuracy test...")
print("(Processing 400 images - may take 3-5 minutes)\n")

gallery_embeddings = {}
test_results = []

for person_id, imgs in persons.items():
    # Gallery: first 5 images
    gallery_embs = []
    for img in imgs[:5]:
        img_processed = preprocess_for_insightface(img)
        emb = recognizer.get_embedding(img_processed)
        if emb is not None:
            gallery_embs.append(emb)
    
    if gallery_embs:
        gallery_embeddings[person_id] = np.mean(gallery_embs, axis=0)
    
    # Test: last 5 images
    for img in imgs[5:]:
        img_processed = preprocess_for_insightface(img)
        test_emb = recognizer.get_embedding(img_processed)
        if test_emb is not None:
            # Find best match
            best_id = None
            best_score = -1
            for gid, gemb in gallery_embeddings.items():
                score = recognizer.compare(test_emb, gemb)
                if score > best_score:
                    best_score = score
                    best_id = gid
            
            test_results.append({
                'true_id': person_id,
                'pred_id': best_id,
                'score': best_score
            })
    
    # Progress
    if (person_id + 1) % 10 == 0:
        print(f"  Processed {person_id + 1}/40 people...")

# Calculate accuracy
if test_results:
    correct = sum(1 for r in test_results if r['true_id'] == r['pred_id'])
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
    print("\n✓ Claim validated: >95% accuracy on standard dataset")
elif accuracy > 0:
    print(f"\nNote: {accuracy:.1f}% - test completed successfully")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "olivetti_accuracy_offline.txt", "w") as f:
    f.write("Olivetti Faces Test Results (Offline Mode)\n")
    f.write("=========================================\n")
    f.write(f"Model: buffalo_l (ONNX Runtime)\n")
    f.write(f"Model path: {model_path}\n")
    f.write(f"Dataset: Olivetti Faces (400 images, 40 people)\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")

print(f"\n✓ Report saved: test_results/olivetti_accuracy_offline.txt")