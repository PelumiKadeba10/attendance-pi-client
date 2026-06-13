#!/usr/bin/env python3
"""
Frontal Face Accuracy Test for attendance-pi-client
Uses existing buffalo_l model files without downloading
"""

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import onnxruntime as ort
from pathlib import Path
import sys
import time

class SimpleFaceRecognizer:
    """Simple face recognizer using your existing buffalo_l ONNX files"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # Load ONNX models
        print("Loading face detection model...")
        det_path = self.model_path / "det_10g.onnx"
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        
        print("Loading face recognition model...")
        rec_path = self.model_path / "w600k_r50.onnx"
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        print("Loading landmark detection model...")
        land_path = self.model_path / "2d106det.onnx"
        self.land_session = ort.InferenceSession(str(land_path), providers=['CPUExecutionProvider'])
        
        # Input/Output names
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_names = [out.name for out in self.det_session.get_outputs()]
        
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name
        
        print("Models loaded successfully!\n")
    
    def detect_faces(self, image):
        """Detect faces in image"""
        # Prepare input (1,3,640,640)
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (640, 640))
        img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_resized = (img_resized - 127.5) / 128.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        # Run detection
        detections = self.det_session.run(self.det_output_names, {self.det_input_name: img_resized})
        
        # Parse detections (simplified - just get bounding boxes)
        boxes = detections[0][0]  # Shape: (n,6) where 6 = [x1,y1,x2,y2,score,class]
        
        # Filter by confidence and scale back to original size
        faces = []
        for box in boxes:
            if box[4] > 0.5:  # Confidence threshold
                x1 = int(box[0] * w / 640)
                y1 = int(box[1] * h / 640)
                x2 = int(box[2] * w / 640)
                y2 = int(box[3] * h / 640)
                
                # Get face crop
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size > 0:
                    # Get embedding
                    emb = self.get_embedding(face_crop)
                    faces.append({
                        'box': [x1, y1, x2, y2],
                        'embedding': emb,
                        'confidence': box[4]
                    })
        
        return faces
    
    def get_embedding(self, face_image):
        """Extract face embedding"""
        # Resize to 112x112 for recognition model
        face_resized = cv2.resize(face_image, (112, 112))
        face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
        face_resized = (face_resized - 127.5) / 128.0
        face_resized = np.expand_dims(face_resized, axis=0)
        
        # Run recognition
        embedding = self.rec_session.run([self.rec_output_name], {self.rec_input_name: face_resized})[0]
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.flatten()
    
    def compare(self, emb1, emb2):
        """Compare two embeddings using cosine similarity"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def test_frontal_accuracy():
    """Test frontal face accuracy using LFW dataset"""
    
    print("="*60)
    print("FRONTAL FACE ACCURACY TEST")
    print("="*60)
    
    # Initialize recognizer with your model
    model_path = Path(__file__).parent / "models" / "buffalo_l"
    recognizer = SimpleFaceRecognizer(model_path)
    
    # Load LFW dataset
    print("\n[1/4] Loading LFW dataset...")
    print("(This downloads ~200MB on first run)")
    lfw_dataset = fetch_lfw_people(min_faces_per_person=50, resize=0.5)
    
    X_images = lfw_dataset.images
    y_labels = lfw_dataset.target
    target_names = lfw_dataset.target_names
    
    # Convert grayscale to BGR
    print("[2/4] Preparing images...")
    X_bgr = []
    for img in X_images:
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        X_bgr.append(img_bgr)
    
    print(f"Loaded {len(X_bgr)} images of {len(target_names)} people")
    
    # Create gallery (one image per person)
    print("\n[3/4] Creating reference gallery...")
    gallery_embeddings = []
    gallery_labels = []
    gallery_names = []
    
    unique_people = np.unique(y_labels)
    
    for person_id in unique_people[:20]:  # Limit to 20 people for faster testing
        # Find images of this person
        person_indices = np.where(y_labels == person_id)[0]
        if len(person_indices) == 0:
            continue
        
        # Use first image as gallery
        idx = person_indices[0]
        img = X_bgr[idx]
        
        faces = recognizer.detect_faces(img)
        if faces:
            gallery_embeddings.append(faces[0]['embedding'])
            gallery_labels.append(person_id)
            gallery_names.append(target_names[person_id])
            print(f"  ✓ Added: {target_names[person_id]}")
    
    print(f"\nGallery created with {len(gallery_embeddings)} people")
    
    # Test on remaining images
    print("\n[4/4] Testing...")
    correct = 0
    total = 0
    
    # Skip gallery images from testing
    gallery_indices = [np.where(y_labels == pid)[0][0] for pid in gallery_labels]
    
    for i, img in enumerate(X_bgr[:500]):  # Test first 500 images for speed
        if i in gallery_indices:
            continue
        
        faces = recognizer.detect_faces(img)
        if not faces:
            continue
        
        test_embedding = faces[0]['embedding']
        true_label = y_labels[i]
        total += 1
        
        # Find best match
        best_similarity = -1
        best_idx = -1
        
        for j, gallery_emb in enumerate(gallery_embeddings):
            similarity = recognizer.compare(test_embedding, gallery_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = j
        
        if best_idx >= 0 and gallery_labels[best_idx] == true_label:
            correct += 1
        
        # Progress indicator
        if total % 50 == 0:
            print(f"  Tested {total} images...")
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total test images:     {total}")
    print(f"Correct recognitions:  {correct}")
    print(f"Incorrect:             {total - correct}")
    print(f"\n✅ FRONTAL FACE ACCURACY: {accuracy:.2f}%")
    
    if accuracy > 95:
        print(f"\n✓ CLAIM VALIDATED: Accuracy >95% ({accuracy:.2f}%)")
    else:
        print(f"\n⚠ CLAIM NOT MET: Expected >95%, got {accuracy:.2f}%")
    
    # Save results
    results_file = Path(__file__).parent / "test_results" / "frontal_accuracy.txt"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Frontal Face Accuracy Test Results\n")
        f.write(f"==================================\n")
        f.write(f"Model: InsightFace buffalo_l\n")
        f.write(f"Model files: {list(model_path.glob('*.onnx'))}\n")
        f.write(f"Dataset: LFW (Labeled Faces in the Wild)\n")
        f.write(f"Test images: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Threshold: >95%\n")
        f.write(f"Result: {'PASS' if accuracy > 95 else 'FAIL'}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return accuracy

def main():
    try:
        accuracy = test_frontal_accuracy()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the virtual environment: source venv/bin/activate")
        print("2. Install missing packages: pip install scikit-learn onnxruntime opencv-python")
        sys.exit(1)

if __name__ == "__main__":
    main()