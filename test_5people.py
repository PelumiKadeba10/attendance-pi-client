#!/usr/bin/env python3
"""
LFW 5-Person Accuracy Test - Using ONNX Runtime Directly
NO DOWNLOAD - Uses your local buffalo_l .onnx files
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

class DirectFaceRecognizer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # Load detection model
        det_path = self.model_path / "det_10g.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")
        
        # Load recognition model
        rec_path = self.model_path / "w600k_r50.onnx"
        if not rec_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_path}")
        
        # Load with CPU only - no network access
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        # Get input/output names
        self.det_input = self.det_session.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.det_session.get_outputs()]
        self.rec_input = self.rec_session.get_inputs()[0].name
        self.rec_output = self.rec_session.get_outputs()[0].name
        
        print(f"✓ Detection model: {det_path.name}")
        print(f"✓ Recognition model: {rec_path.name}")
    
    def get_embedding(self, image):
        """Extract face embedding from image"""
        try:
            h, w = image.shape[:2]
            
            # Resize for detection (640x640)
            img_resized = cv2.resize(image, (640, 640))
            img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
            img_resized = (img_resized - 127.5) / 128.0
            img_resized = np.expand_dims(img_resized, axis=0)
            
            # Run detection
            detections = self.det_session.run(self.det_outputs, {self.det_input: img_resized})
            
            # Parse boxes
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
                    
                    if score > 0.5 and score > best_score:
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
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    
                    return embedding.flatten()
            
            return None
            
        except Exception as e:
            return None
    
    def compare(self, emb1, emb2):
        """Cosine similarity"""
        if emb1 is None or emb2 is None:
            return -1
        return float(np.dot(emb1, emb2))

print("="*60)
print("LFW 5-PERSON ACCURACY TEST (OFFLINE)")
print("="*60)

# Load model
print("\n[1/4] Loading model from local files...")
model_path = Path(__file__).parent / "models" / "buffalo_l"

if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    print("Checking alternative locations...")
    
    # Try other possible locations
    alt_paths = [
        Path(__file__).parent / "models",
        Path(__file__).parent / "models/buffalo_l",
        Path.home() / "attendance-pi-client/models/buffalo_l",
    ]
    
    for alt in alt_paths:
        if alt.exists() and (alt / "det_10g.onnx").exists():
            model_path = alt
            print(f"✓ Found model at: {model_path}")
            break
    else:
        print("❌ Could not find buffalo_l model files")
        print("Make sure you have:")
        print("  - det_10g.onnx")
        print("  - w600k_r50.onnx")
        sys.exit(1)

recognizer = DirectFaceRecognizer(model_path)

# Load dataset
print("\n[2/4] Loading LFW 5-person subset...")
dataset_path = Path(__file__).parent / "datasets" / "lfw_5people"

if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    print("\nPlease transfer lfw_5people.zip to datasets/ and unzip it:")
    print("  cd ~/attendance-pi-client/datasets")
    print("  unzip lfw_5people.zip")
    sys.exit(1)

# Load images for each person
persons = {}
for person_dir in dataset_path.iterdir():
    if person_dir.is_dir():
        images = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        if images:
            persons[person_dir.name] = images
            print(f"  ✓ {person_dir.name}: {len(images)} images")

print(f"\n✓ Loaded {len(persons)} people")

# Test parameters
GALLERY_SIZE = 5
TEST_SIZE = 15  # Test on first 15 images after gallery

print(f"\n[3/4] Building gallery (first {GALLERY_SIZE} images per person)...")

# Build gallery embeddings
gallery = {}
for person_name, images in persons.items():
    embeddings = []
    for i in range(min(GALLERY_SIZE, len(images))):
        emb = recognizer.get_embedding(images[i])
        if emb is not None:
            embeddings.append(emb)
    
    if embeddings:
        gallery[person_name] = np.mean(embeddings, axis=0)
        print(f"  ✓ {person_name}: {len(embeddings)}/{GALLERY_SIZE} faces detected")
    else:
        print(f"  ✗ {person_name}: No faces detected!")

if len(gallery) < 2:
    print("\n❌ Not enough faces detected. Trying with different parameters...")
    sys.exit(1)

print(f"\n[4/4] Running verification tests (next {TEST_SIZE} images per person)...")

# Test remaining images
correct = 0
total = 0

for person_name, images in persons.items():
    if person_name not in gallery:
        continue
    
    for i in range(GALLERY_SIZE, min(GALLERY_SIZE + TEST_SIZE, len(images))):
        test_emb = recognizer.get_embedding(images[i])
        if test_emb is not None:
            total += 1
            
            # Find best match
            best_match = None
            best_score = -1
            for gallery_name, gallery_emb in gallery.items():
                score = recognizer.compare(test_emb, gallery_emb)
                if score > best_score:
                    best_score = score
                    best_match = gallery_name
            
            if best_match == person_name:
                correct += 1

# Calculate accuracy
accuracy = (correct / total * 100) if total > 0 else 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total tests:        {total}")
print(f"Correct:            {correct}")
print(f"Failed:             {total - correct}")
print(f"\n✅ ACCURACY: {accuracy:.1f}%")

if accuracy > 95:
    print("\n✓ CLAIM VALIDATED: >95% accuracy on LFW subset")
elif accuracy > 85:
    print(f"\n⚠ Good result: {accuracy:.1f}% - Close to target")
else:
    print(f"\n⚠ Accuracy lower than expected: {accuracy:.1f}%")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "lfw_onnx_results.txt", "w") as f:
    f.write("LFW 5-Person Accuracy Test (ONNX Runtime)\n")
    f.write("=========================================\n\n")
    f.write(f"Model: buffalo_l (ONNX Runtime, offline)\n")
    f.write(f"Model path: {model_path}\n")
    f.write(f"Test subjects: {', '.join(gallery.keys())}\n")
    f.write(f"Gallery images per person: {GALLERY_SIZE}\n")
    f.write(f"Test images per person: {TEST_SIZE}\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")

print(f"\n✓ Report saved: test_results/lfw_onnx_results.txt")