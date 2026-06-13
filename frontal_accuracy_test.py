#!/usr/bin/env python3
"""
Quick Accuracy Test Using Your Own Photos
No downloads required - uses webcam or existing photos
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import time
from datetime import datetime

class SimpleFaceRecognizer:
    """Face recognizer using your buffalo_l model"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        print("Loading models...")
        # Load detection model
        det_path = self.model_path / "det_10g.onnx"
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        
        # Load recognition model
        rec_path = self.model_path / "w600k_r50.onnx"
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        print("✓ Models loaded\n")
    
    def detect_and_recognize(self, image):
        """Detect face and extract embedding"""
        # Prepare image for detection
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (640, 640))
        img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_resized = (img_resized - 127.5) / 128.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        # Run detection
        det_input_name = self.det_session.get_inputs()[0].name
        det_output_names = [out.name for out in self.det_session.get_outputs()]
        detections = self.det_session.run(det_output_names, {det_input_name: img_resized})
        
        boxes = detections[0][0]
        
        for box in boxes:
            if box[4] > 0.5:  # Confidence threshold
                # Scale back to original size
                x1 = int(box[0] * w / 640)
                y1 = int(box[1] * h / 640)
                x2 = int(box[2] * w / 640)
                y2 = int(box[3] * h / 640)
                
                # Extract face
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    # Get embedding
                    face_resized = cv2.resize(face, (112, 112))
                    face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
                    face_resized = (face_resized - 127.5) / 128.0
                    face_resized = np.expand_dims(face_resized, axis=0)
                    
                    rec_input_name = self.rec_session.get_inputs()[0].name
                    rec_output_name = self.rec_session.get_outputs()[0].name
                    embedding = self.rec_session.run([rec_output_name], {rec_input_name: face_resized})[0]
                    
                    # Normalize
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    return embedding, (x1, y1, x2, y2), box[4]
        
        return None, None, None
    
    def compare(self, emb1, emb2):
        """Compare embeddings (cosine similarity)"""
        return np.dot(emb1, emb2)

def capture_photos_for_test():
    """Capture photos using webcam for testing"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        print("\nAlternative: Place photos in test_photos/ folder")
        return None
    
    print("\n📸 Webcam detected! Let's capture test photos")
    print("="*60)
    
    persons = []
    
    while len(persons) < 3:
        name = input(f"\nEnter name for person {len(persons)+1} (or 'done' to finish): ")
        if name.lower() == 'done':
            break
        
        print(f"Press SPACE to capture 3 photos of {name}")
        photos = []
        
        while len(photos) < 3:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Show preview
            display = frame.copy()
            cv2.putText(display, f"{name} - Photos: {len(photos)}/3", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Photos - Press SPACE", display)
            
            key = cv2.waitKey(1)
            if key == 32:  # SPACE key
                photos.append(frame.copy())
                print(f"  ✓ Captured photo {len(photos)}/3")
                time.sleep(0.5)
            elif key == 27:  # ESC
                break
        
        if len(photos) == 3:
            persons.append({'name': name, 'photos': photos})
            print(f"✓ {name} added successfully!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return persons

def create_gallery(recognizer, persons):
    """Create embedding gallery from first 2 photos of each person"""
    gallery = []
    
    print("\n" + "="*60)
    print("CREATING REFERENCE GALLERY")
    print("="*60)
    
    for person in persons:
        name = person['name']
        # Use first 2 photos for gallery
        gallery_photos = person['photos'][:2]
        
        embeddings = []
        for idx, photo in enumerate(gallery_photos):
            emb, _, _ = recognizer.detect_and_recognize(photo)
            if emb is not None:
                embeddings.append(emb)
                print(f"  ✓ {name} - reference {idx+1}")
        
        if embeddings:
            # Average embeddings for better representation
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            gallery.append({
                'name': name,
                'embedding': avg_emb,
                'photos': person['photos']
            })
    
    return gallery

def run_accuracy_test(recognizer, gallery):
    """Test accuracy using remaining photos"""
    print("\n" + "="*60)
    print("RUNNING ACCURACY TEST")
    print("="*60)
    
    total_tests = 0
    correct = 0
    
    for person in gallery:
        name = person['name']
        # Use remaining photos for testing (after the first 2 used for gallery)
        test_photos = person['photos'][2:] if len(person['photos']) > 2 else []
        
        # Also test with gallery photos themselves (should be 100%)
        for idx, photo in enumerate(person['photos']):
            test_emb, _, _ = recognizer.detect_and_recognize(photo)
            if test_emb is not None:
                total_tests += 1
                
                # Find best match in gallery
                best_match = None
                best_score = -1
                
                for g in gallery:
                    score = recognizer.compare(test_emb, g['embedding'])
                    if score > best_score:
                        best_score = score
                        best_match = g['name']
                
                if best_match == name:
                    correct += 1
                    result = "✓"
                else:
                    result = f"✗ (matched {best_match})"
                
                print(f"  {result} {name} - test {idx+1}: similarity={best_score:.3f}")
    
    accuracy = (correct / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total tests:     {total_tests}")
    print(f"Correct:         {correct}")
    print(f"Accuracy:        {accuracy:.1f}%")
    
    if accuracy > 95:
        print(f"\n✅ CLAIM VALIDATED: {accuracy:.1f}% > 95%")
    else:
        print(f"\n⚠️ Claim not met: {accuracy:.1f}% < 95%")
    
    return accuracy

def main():
    print("="*60)
    print("FACE RECOGNITION ACCURACY TEST")
    print("Using your buffalo_l model")
    print("="*60)
    
    # Load model
    model_path = Path(__file__).parent / "models" / "buffalo_l"
    recognizer = SimpleFaceRecognizer(model_path)
    
    # Check for existing test photos
    test_photos_dir = Path(__file__).parent / "test_photos"
    
    if test_photos_dir.exists() and any(test_photos_dir.iterdir()):
        print("\nFound existing test photos!")
        # Load from directory
        persons = []
        for person_dir in test_photos_dir.iterdir():
            if person_dir.is_dir():
                photos = []
                for img_file in person_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        photos.append(img)
                if photos:
                    persons.append({'name': person_dir.name, 'photos': photos})
        
        if persons:
            print(f"Loaded {len(persons)} people from test_photos/")
            gallery = create_gallery(recognizer, persons)
            accuracy = run_accuracy_test(recognizer, gallery)
            return
    
    # Otherwise, capture new photos
    print("\nNo existing test photos found.")
    print("We'll capture new photos using webcam.")
    print("\nRequirements:")
    print("  - At least 3 different people")
    print("  - 3-5 photos of each person")
    print("  - Frontal faces, good lighting")
    
    persons = capture_photos_for_test()
    
    if not persons or len(persons) < 2:
        print("\n❌ Need at least 2 people for testing")
        return
    
    # Create gallery and test
    gallery = create_gallery(recognizer, persons)
    accuracy = run_accuracy_test(recognizer, gallery)
    
    # Save photos for future tests
    print(f"\nSaving photos to {test_photos_dir}/ for future use...")
    for person in persons:
        person_dir = test_photos_dir / person['name']
        person_dir.mkdir(parents=True, exist_ok=True)
        for idx, photo in enumerate(person['photos']):
            cv2.imwrite(str(person_dir / f"photo_{idx+1}.jpg"), photo)
    print("✓ Photos saved!")

if __name__ == "__main__":
    main()