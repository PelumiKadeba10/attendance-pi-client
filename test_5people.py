#!/usr/bin/env python3
"""
Accuracy Test with 5 LFW Subjects
Uses your local buffalo_l model
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path
import sys

print("="*60)
print("LFW 5-PERSON ACCURACY TEST")
print("="*60)

# Load model from correct local path
print("\n[1/3] Loading InsightFace model...")
model_root = Path(__file__).parent / "models"

# Check if buffalo_l exists
buffalo_path = model_root / "buffalo_l"
if not buffalo_path.exists():
    print(f"❌ Model not found at {buffalo_path}")
    print("Current models folder contents:")
    for p in model_root.iterdir():
        print(f"  - {p.name}")
    sys.exit(1)

print(f"✓ Found model at: {buffalo_path}")

# Initialize with correct root (parent of buffalo_l)
app = FaceAnalysis(name='buffalo_l', root=str(model_root))
app.prepare(ctx_id=0, det_size=(320, 320))
print("✓ Model loaded")

# Load dataset
print("\n[2/3] Loading 5-person LFW subset...")
dataset_path = Path(__file__).parent / "datasets" / "lfw_5people"

if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    print("Please run: unzip datasets/lfw_5people.zip")
    exit(1)

# Load images for each person
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

# Test parameters
GALLERY_SIZE = 5  # Use first 5 images as reference
TEST_SIZE = 20    # Test on next 20 images

print(f"\n[3/3] Running accuracy test...")
print(f"  Gallery: {GALLERY_SIZE} images per person")
print(f"  Testing: {TEST_SIZE} images per person")
print()

# Build gallery embeddings
gallery = {}
for person_name, images in persons.items():
    embeddings = []
    for i in range(min(GALLERY_SIZE, len(images))):
        faces = app.get(images[i])
        if faces:
            embeddings.append(faces[0].embedding)
    
    if embeddings:
        gallery[person_name] = np.mean(embeddings, axis=0)
        print(f"  ✓ Gallery: {person_name} ({len(embeddings)} images)")

# Test remaining images
correct = 0
total = 0
results = []

for person_name, images in persons.items():
    for i in range(GALLERY_SIZE, min(GALLERY_SIZE + TEST_SIZE, len(images))):
        faces = app.get(images[i])
        if faces:
            test_emb = faces[0].embedding
            total += 1
            
            # Find best match
            best_match = None
            best_score = -1
            for gallery_name, gallery_emb in gallery.items():
                score = np.dot(test_emb, gallery_emb)
                if score > best_score:
                    best_score = score
                    best_match = gallery_name
            
            if best_match == person_name:
                correct += 1
                results.append("✓")
            else:
                results.append(f"✗ (matched {best_match})")

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
    print("\n✓ CLAIM VALIDATED: >95% accuracy")
elif accuracy > 90:
    print(f"\n⚠ Good: {accuracy:.1f}% - close to 95% target")
else:
    print(f"\n⚠ Below target: {accuracy:.1f}%")

# Save report
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "lfw_5people_results.txt", "w") as f:
    f.write("LFW 5-Person Accuracy Test Results\n")
    f.write("==================================\n\n")
    f.write(f"Test subjects: {', '.join(persons.keys())}\n")
    f.write(f"Gallery images per person: {GALLERY_SIZE}\n")
    f.write(f"Test images per person: {TEST_SIZE}\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")
    f.write(f"Threshold: >95%\n")
    f.write(f"Result: {'PASS' if accuracy > 95 else 'REVIEW'}\n")

print(f"\n✓ Report saved: test_results/lfw_5people_results.txt")