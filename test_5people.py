#!/usr/bin/env python3
"""
LFW 5-Person Accuracy Test - HIGH RESOLUTION VERSION
For use with full-res images (250x250 pixels)
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import time

class DirectFaceRecognizer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        det_path = self.model_path / "det_10g.onnx"
        rec_path = self.model_path / "w600k_r50.onnx"
        
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")
        if not rec_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_path}")
        
        print("Loading models...")
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        self.det_input = self.det_session.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.det_session.get_outputs()]
        self.rec_input = self.rec_session.get_inputs()[0].name
        self.rec_output = self.rec_session.get_outputs()[0].name
        
        print(f"✓ Models loaded\n")
    
    def get_embedding(self, image):
        """Extract face embedding from high-res image"""
        try:
            h, w = image.shape[:2]
            
            # Resize for detection (640x640)
            img_resized = cv2.resize(image, (640, 640))
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
                    
                    if score > 0.5 and score > best_score:
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
                
                face = image[y1:y2, x1:x2]
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

print("="*60)
print("LFW HIGH-RESOLUTION ACCURACY TEST")
print("="*60)

# Load model
print("\n[1/4] Loading model...")
model_path = Path(__file__).parent / "models" / "buffalo_l"
recognizer = DirectFaceRecognizer(model_path)

# Load dataset - try different possible paths
print("\n[2/4] Loading LFW dataset...")

possible_paths = [
    Path(__file__).parent / "datasets" ,
    Path(__file__).parent / "datasets" / "lfw_5people_fullres",
    Path(__file__).parent / "datasets" / "lfw_5people",
]

dataset_path = None
for path in possible_paths:
    if path.exists():
        dataset_path = path
        print(f"✓ Found dataset at: {dataset_path}")
        break

if dataset_path is None:
    print("❌ Dataset not found. Tried:")
    for p in possible_paths:
        print(f"  - {p}")
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

# Check image resolution
sample_img = list(persons.values())[0][0]
print(f"\n✓ Image resolution: {sample_img.shape[1]}x{sample_img.shape[0]} pixels")
if sample_img.shape[0] < 100:
    print("  ⚠ Warning: Images are small. Detection may still fail.")
else:
    print("  ✓ High resolution - perfect for detection!")

# Test face detection on first few images
print("\n[3/4] Testing face detection...")
detection_stats = {}

for person_name, images in list(persons.items())[:5]:  # Test first 5 people
    detected = 0
    for i in range(min(5, len(images))):
        emb = recognizer.get_embedding(images[i])
        if emb is not None:
            detected += 1
    detection_stats[person_name] = detected
    print(f"  {person_name}: {detected}/5 faces detected")

total_detected = sum(detection_stats.values())
if total_detected < 10:
    print("\n⚠️ Low face detection rate. Check if images are high-res.")
    print(f"   Detected: {total_detected}/{(len(detection_stats)*5)}")
    sys.exit(1)

# Build gallery and test
print("\n[4/4] Building gallery and running accuracy test...")
print("(This may take 5-10 minutes on Raspberry Pi)")

GALLERY_SIZE = 5
TEST_SIZE = 20  # Test on 20 images per person

gallery = {}
test_results = []
total_processed = 0

for person_name, images in persons.items():
    # Build gallery from first GALLERY_SIZE images
    gallery_embs = []
    for i in range(min(GALLERY_SIZE, len(images))):
        emb = recognizer.get_embedding(images[i])
        if emb is not None:
            gallery_embs.append(emb)
    
    if gallery_embs:
        gallery[person_name] = np.mean(gallery_embs, axis=0)
        print(f"  ✓ {person_name}: gallery built ({len(gallery_embs)} images)")
        
        # Test on next TEST_SIZE images
        for i in range(GALLERY_SIZE, min(GALLERY_SIZE + TEST_SIZE, len(images))):
            test_emb = recognizer.get_embedding(images[i])
            if test_emb is not None:
                total_processed += 1
                
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
print(f"People in gallery:  {len(gallery)}")
print(f"Total tests:        {total}")
print(f"Correct:            {correct}")
print(f"Failed:             {total - correct}")
print(f"Average similarity: {np.mean([r['score'] for r in test_results]):.3f}")
print(f"\n✅ ACCURACY: {accuracy:.1f}%")

if accuracy > 95:
    print("\n✓ CLAIM VALIDATED: >95% accuracy on LFW subset")
elif accuracy > 90:
    print(f"\n⚠ Good: {accuracy:.1f}% - Close to 95% target")
elif accuracy > 0:
    print(f"\n⚠ Below target: {accuracy:.1f}%")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "lfw_highres_results.txt", "w") as f:
    f.write("LFW High-Resolution Accuracy Test Results\n")
    f.write("=========================================\n\n")
    f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: buffalo_l (ONNX Runtime)\n")
    f.write(f"Dataset: LFW funneled (high-res)\n")
    f.write(f"People tested: {list(gallery.keys())}\n")
    f.write(f"Gallery images per person: {GALLERY_SIZE}\n")
    f.write(f"Test images per person: {TEST_SIZE}\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")

print(f"\n✓ Report saved: test_results/lfw_highres_results.txt")