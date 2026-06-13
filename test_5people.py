#!/usr/bin/env python3
"""
LFW Direct Recognition Test - Skip Detection, Use Pre-cropped Faces
LFW images are already aligned and cropped, so we can feed them directly
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import time

class DirectRecognizer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # Only need recognition model (no detection needed!)
        rec_path = self.model_path / "w600k_r50.onnx"
        
        if not rec_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_path}")
        
        print("Loading recognition model (no detection needed)...")
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        self.rec_input = self.rec_session.get_inputs()[0].name
        self.rec_output = self.rec_session.get_outputs()[0].name
        
        print(f"✓ Recognition model loaded\n")
    
    def get_embedding_from_face(self, face_image):
        """Extract embedding directly from pre-cropped face"""
        try:
            # LFW images are already cropped faces, just resize to 112x112
            face_resized = cv2.resize(face_image, (112, 112))
            face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
            face_resized = (face_resized - 127.5) / 128.0
            face_resized = np.expand_dims(face_resized, axis=0)
            
            embedding = self.rec_session.run([self.rec_output], {self.rec_input: face_resized})[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.flatten()
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def compare(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return -1
        return float(np.dot(emb1, emb2))

print("="*60)
print("LFW DIRECT RECOGNITION TEST")
print("(No face detection - using pre-cropped images)")
print("="*60)

# Load model
print("\n[1/3] Loading recognition model...")
model_path = Path(__file__).parent / "models" / "buffalo_l"
recognizer = DirectRecognizer(model_path)

# Load dataset
print("\n[2/3] Loading LFW images...")
dataset_path = Path(__file__).parent / "datasets"

if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    sys.exit(1)

# Find the person folders
persons = {}
for person_dir in dataset_path.iterdir():
    if person_dir.is_dir() and person_dir.name in ["Colin_Powell", "Ariel_Sharon", "George_W_Bush", "Donald_Rumsfeld", "Tony_Blair"]:
        images = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        if images:
            persons[person_dir.name] = images
            print(f"  ✓ {person_dir.name}: {len(images)} images")

if len(persons) < 3:
    print(f"❌ Only found {len(persons)} people. Need at least 3.")
    # Show what's available
    print("\nAvailable folders:")
    for d in dataset_path.iterdir():
        if d.is_dir():
            print(f"  - {d.name}")
    sys.exit(1)

print(f"\n✓ Loaded {len(persons)} people")

# Test parameters
GALLERY_SIZE = 5
TEST_SIZE = 30  # Test on 30 images per person

print(f"\n[3/3] Running accuracy test...")
print(f"  Gallery: {GALLERY_SIZE} images per person")
print(f"  Testing: {TEST_SIZE} images per person")
print()

# Build gallery (use first GALLERY_SIZE images per person)
gallery = {}
gallery_counts = {}

for person_name, images in persons.items():
    embeddings = []
    for i in range(min(GALLERY_SIZE, len(images))):
        # Convert grayscale to BGR if needed (LFW is sometimes grayscale)
        img = images[i]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        emb = recognizer.get_embedding_from_face(img)
        if emb is not None:
            embeddings.append(emb)
    
    if embeddings:
        gallery[person_name] = np.mean(embeddings, axis=0)
        gallery_counts[person_name] = len(embeddings)
        print(f"  ✓ {person_name}: gallery built ({len(embeddings)}/{GALLERY_SIZE} images)")

if len(gallery) < 2:
    print("\n❌ Failed to build gallery. Check if images load correctly.")
    sys.exit(1)

# Test on remaining images
correct = 0
total = 0
confusion = {name: {'correct': 0, 'total': 0} for name in gallery.keys()}

for person_name, images in persons.items():
    # Test from GALLERY_SIZE to GALLERY_SIZE+TEST_SIZE
    for i in range(GALLERY_SIZE, min(GALLERY_SIZE + TEST_SIZE, len(images))):
        img = images[i]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        test_emb = recognizer.get_embedding_from_face(img)
        if test_emb is not None:
            total += 1
            confusion[person_name]['total'] += 1
            
            # Find best match
            best_match = None
            best_score = -1
            for gname, gemb in gallery.items():
                score = recognizer.compare(test_emb, gemb)
                if score > best_score:
                    best_score = score
                    best_match = gname
            
            if best_match == person_name:
                correct += 1
                confusion[person_name]['correct'] += 1

accuracy = (correct / total * 100) if total > 0 else 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total tests:        {total}")
print(f"Correct:            {correct}")
print(f"Failed:             {total - correct}")
print(f"\n✅ ACCURACY: {accuracy:.1f}%")

print("\nPer-person accuracy:")
for name in gallery.keys():
    c = confusion[name]['correct']
    t = confusion[name]['total']
    if t > 0:
        print(f"  {name}: {c}/{t} ({c/t*100:.1f}%)")

if accuracy > 95:
    print("\n✓ CLAIM VALIDATED: >95% accuracy")
elif accuracy > 90:
    print("\n⚠ Good: >90% accuracy")
else:
    print("\n⚠ Needs investigation")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "lfw_direct_results.txt", "w") as f:
    f.write("LFW Direct Recognition Test Results\n")
    f.write("===================================\n\n")
    f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: buffalo_l (recognition only, no detection)\n")
    f.write(f"Method: Pre-cropped faces fed directly to recognition model\n")
    f.write(f"Gallery size: {GALLERY_SIZE} images/person\n")
    f.write(f"Test size: {TEST_SIZE} images/person\n")
    f.write(f"People tested: {list(gallery.keys())}\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")

print(f"\n✓ Report saved: test_results/lfw_direct_results.txt")