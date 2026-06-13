#!/usr/bin/env python3
"""
Frontal Face Accuracy Test for attendance-pi-client
Tests InsightFace buffalo_l model on LFW dataset
"""

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from insightface.app import FaceAnalysis
import os
import sys
from pathlib import Path

# Add your project to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_insightface_model(model_path=None):
    """Load the buffalo_l model from your existing models directory"""
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "buffalo_l"
    
    print(f"Loading model from: {model_path}")
    
    # Initialize InsightFace with your local model path
    app = FaceAnalysis(name='buffalo_l', root=str(Path(__file__).parent / "models"))
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU
    
    print("Model loaded successfully!")
    return app

def test_frontal_accuracy(app):
    """Test frontal face accuracy using LFW dataset"""
    
    print("\n" + "="*60)
    print("FRONTAL FACE ACCURACY TEST")
    print("="*60)
    
    # Download LFW dataset (only need to do this once)
    print("\n[1/4] Loading LFW dataset...")
    print("(This downloads ~200MB on first run, may take a few minutes)")
    lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    X_images = lfw_dataset.images  # Grayscale images
    y_labels = lfw_dataset.target
    target_names = lfw_dataset.target_names
    
    # Convert grayscale to BGR (InsightFace expects BGR)
    print("[2/4] Converting images to BGR format...")
    X_bgr = []
    for img in X_images:
        # Convert grayscale (0-1 float) to uint8 BGR
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        X_bgr.append(img_bgr)
    X_bgr = np.array(X_bgr)
    
    print(f"Dataset loaded: {len(X_bgr)} images, {len(target_names)} people")
    
    # Create gallery (one reference image per person)
    print("\n[3/4] Creating reference gallery...")
    gallery_embeddings = []
    gallery_labels = []
    gallery_names = []
    
    for person_id in np.unique(y_labels):
        # Get first image of this person
        idx = np.where(y_labels == person_id)[0][0]
        img = X_bgr[idx]
        
        # Detect face and get embedding
        faces = app.get(img)
        if faces and len(faces) > 0:
            embedding = faces[0].embedding
            gallery_embeddings.append(embedding)
            gallery_labels.append(person_id)
            gallery_names.append(target_names[person_id])
            print(f"  ✓ Added: {target_names[person_id]}")
        else:
            print(f"  ✗ No face detected for: {target_names[person_id]}")
    
    gallery_embeddings = np.array(gallery_embeddings)
    print(f"Gallery created with {len(gallery_embeddings)} people")
    
    # Test on remaining images
    print("\n[4/4] Testing on all images...")
    correct = 0
    total = 0
    failed_images = []
    
    # Skip the gallery images from testing
    gallery_indices = [np.where(y_labels == pid)[0][0] for pid in np.unique(y_labels)]
    
    for i, img in enumerate(X_bgr):
        # Skip gallery images
        if i in gallery_indices:
            continue
        
        # Detect face
        faces = app.get(img)
        if not faces or len(faces) == 0:
            continue
        
        test_embedding = faces[0].embedding
        true_label = y_labels[i]
        total += 1
        
        # Find best match in gallery (cosine similarity)
        similarities = np.dot(gallery_embeddings, test_embedding)
        best_idx = np.argmax(similarities)
        predicted_label = gallery_labels[best_idx]
        
        if predicted_label == true_label:
            correct += 1
        else:
            failed_images.append({
                'index': i,
                'true_name': target_names[true_label],
                'predicted_name': target_names[gallery_labels[best_idx]],
                'similarity': similarities[best_idx]
            })
        
        # Show progress every 100 images
        if total % 100 == 0:
            print(f"  Progress: {total} images tested...")
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total test images:     {total}")
    print(f"Correct recognitions:  {correct}")
    print(f"Incorrect recognitions: {total - correct}")
    print(f"\n✅ FRONTAL FACE ACCURACY: {accuracy:.2f}%")
    
    if accuracy > 95:
        print(f"\n✓ CLAIM VALIDATED: Accuracy >95% ({accuracy:.2f}%)")
    else:
        print(f"\n⚠ CLAIM NOT MET: Expected >95%, got {accuracy:.2f}%")
    
    # Show some failures (first 5)
    if failed_images and len(failed_images) > 0:
        print(f"\n--- Sample of incorrect matches (first 5 of {len(failed_images)}) ---")
        for j, fail in enumerate(failed_images[:5]):
            print(f"  {j+1}. {fail['true_name']} → misidentified as {fail['predicted_name']}")
    
    return accuracy

def main():
    print("Attendance-Pi: Frontal Face Accuracy Test")
    print("Testing InsightFace buffalo_l model\n")
    
    # Load the model
    app = load_insightface_model()
    
    # Run the test
    accuracy = test_frontal_accuracy(app)
    
    # Save results to file
    results_file = Path(__file__).parent / "test_results" / "frontal_accuracy.txt"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Frontal Face Accuracy Test Results\n")
        f.write(f"==================================\n")
        f.write(f"Model: buffalo_l\n")
        f.write(f"Dataset: LFW (Labeled Faces in the Wild)\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Threshold: >95%\n")
        f.write(f"Result: {'PASS' if accuracy > 95 else 'FAIL'}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()