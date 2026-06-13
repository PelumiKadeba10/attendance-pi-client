#!/usr/bin/env python3
"""
Quick accuracy test with Olivetti Faces (2.5 MB dataset)
Perfect for Raspberry Pi!
Uses buffalo_l model from local models folder
"""

import numpy as np
import cv2
from sklearn.datasets import fetch_olivetti_faces
from insightface.app import FaceAnalysis
from pathlib import Path

print("="*60)
print("OLIVETTI FACES ACCURACY TEST")
print("="*60)

# Load model from local path
print("\n[1/3] Loading InsightFace from local models folder...")
model_path = Path(__file__).parent / "models"

# Initialize FaceAnalysis with local model path
app = FaceAnalysis(name='buffalo_l', root=str(model_path))
app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller det_size for Raspberry Pi

print(f"✓ Model loaded from: {model_path / 'buffalo_l'}")

# Load dataset (2.5 MB)
print("\n[2/3] Loading Olivetti dataset...")
print("(Downloading 2.5 MB from the internet...)")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces.images  # Shape: (400, 64, 64)
targets = faces.target  # Person ID: 0-39

print(f"✓ Loaded {len(images)} images of 40 people (10 each)")

# Group images by person
persons = {}
for i, (img, person_id) in enumerate(zip(images, targets)):
    if person_id not in persons:
        persons[person_id] = []
    persons[person_id].append(img)

# Test: Use first 5 images as gallery, last 5 as test
print("\n[3/3] Running accuracy test...")
print("(This may take 2-3 minutes on Raspberry Pi)")

gallery_embeddings = {}
test_results = []

for person_id, imgs in persons.items():
    # Gallery: first 5 images
    gallery_embs = []
    for img in imgs[:5]:
        # Convert grayscale to BGR (InsightFace expects color)
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        
        faces_detected = app.get(img_bgr)
        if faces_detected:
            gallery_embs.append(faces_detected[0].embedding)
    
    if gallery_embs:
        gallery_embeddings[person_id] = np.mean(gallery_embs, axis=0)
    
    # Test: last 5 images
    for img in imgs[5:]:
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        
        faces_detected = app.get(img_bgr)
        if faces_detected:
            test_emb = faces_detected[0].embedding
            
            # Find best match
            best_id = None
            best_score = -1
            for gid, gemb in gallery_embeddings.items():
                score = np.dot(test_emb, gemb)
                if score > best_score:
                    best_score = score
                    best_id = gid
            
            test_results.append({
                'true_id': person_id,
                'pred_id': best_id,
                'score': best_score
            })
    
    # Print progress every 10 people
    if (person_id + 1) % 10 == 0:
        print(f"  Processed {person_id + 1}/40 people...")

# Calculate accuracy
correct = sum(1 for r in test_results if r['true_id'] == r['pred_id'])
total = len(test_results)
accuracy = (correct / total * 100) if total > 0 else 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total tests:        {total}")
print(f"Correct:            {correct}")
print(f"Failed:             {total - correct}")
print(f"\n✅ ACCURACY: {accuracy:.1f}%")

if accuracy > 95:
    print("\n✓ Claim validated: >95% accuracy on standard dataset")
elif accuracy > 90:
    print(f"\n⚠ Good but below 95%: {accuracy:.1f}% - adjust similarity threshold")
else:
    print(f"\n⚠ Lower than expected: {accuracy:.1f}% - check model loading")

# Save results
report_dir = Path("test_results")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "olivetti_accuracy.txt", "w") as f:
    f.write("Olivetti Faces Test Results\n")
    f.write("==========================\n")
    f.write(f"Model: buffalo_l (from {model_path})\n")
    f.write(f"Dataset: Olivetti Faces (400 images, 40 people)\n")
    f.write(f"Test method: 5 gallery images, 5 test images per person\n")
    f.write(f"Total tests: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")
    f.write(f"Threshold for pass: >95%\n")
    f.write(f"Result: {'PASS' if accuracy > 95 else 'REVIEW'}")

print(f"\n✓ Report saved: test_results/olivetti_accuracy.txt")

# Show where model was loaded from
print(f"\n✓ Model location: {model_path / 'buffalo_l'}")