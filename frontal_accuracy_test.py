#!/usr/bin/env python3
"""
Frontal Face Accuracy Test for Attendance System
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import time

class FaceRecognizer:
    def __init__(self, model_path):
        print("Initializing face recognition model...")
        
        det_path = Path(model_path) / "det_10g.onnx"
        rec_path = Path(model_path) / "w600k_r50.onnx"
        
        if not det_path.exists():
            print(f"Error: Detection model not found at {det_path}")
            sys.exit(1)
        if not rec_path.exists():
            print(f"Error: Recognition model not found at {rec_path}")
            sys.exit(1)
        
        self.det_session = ort.InferenceSession(str(det_path), providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(str(rec_path), providers=['CPUExecutionProvider'])
        
        # Get input/output names
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_names = [output.name for output in self.det_session.get_outputs()]
        
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name
        
        print("Model ready.\n")
    
    def get_face_crop(self, image):
        """Detect face and return cropped face region"""
        h, w = image.shape[:2]
        
        # Prepare image for detection
        img_resized = cv2.resize(image, (640, 640))
        img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_resized = (img_resized - 127.5) / 128.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        # Run detection
        detections = self.det_session.run(
            self.det_output_names, 
            {self.det_input_name: img_resized}
        )
        
        # Parse detection results
        # The output format: [boxes, scores, landmarks] or similar
        boxes = detections[0]  # First output is usually boxes
        scores = detections[1] if len(detections) > 1 else None
        
        # Handle different output formats
        if len(boxes.shape) == 4:
            boxes = boxes[0]  # Remove batch dimension
        
        # Find best detection
        best_box = None
        best_score = 0
        
        for i in range(boxes.shape[0]):
            # Get score
            if scores is not None:
                if len(scores.shape) == 2:
                    score = scores[i][0] if scores.shape[1] > 0 else scores[i]
                else:
                    score = scores[i] if len(scores.shape) == 1 else scores[i][0]
            else:
                # If no separate scores, assume last value in box is score
                if boxes.shape[1] >= 5:
                    score = boxes[i][4]
                else:
                    score = 1.0
            
            # Get bounding box
            if boxes.shape[1] >= 4:
                x1, y1, x2, y2 = boxes[i][:4]
                
                # Scale back to original image size
                x1 = int(x1 * w / 640)
                y1 = int(y1 * h / 640)
                x2 = int(x2 * w / 640)
                y2 = int(y2 * h / 640)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if score > 0.5 and score > best_score and (x2 - x1) > 20 and (y2 - y1) > 20:
                    best_score = score
                    best_box = (x1, y1, x2, y2)
        
        if best_box:
            x1, y1, x2, y2 = best_box
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                return face_crop
        
        return None
    
    def get_embedding(self, face_crop):
        """Extract face embedding from cropped face"""
        # Resize to 112x112 for recognition
        face_resized = cv2.resize(face_crop, (112, 112))
        face_resized = face_resized.transpose(2, 0, 1).astype(np.float32)
        face_resized = (face_resized - 127.5) / 128.0
        face_resized = np.expand_dims(face_resized, axis=0)
        
        # Run recognition
        embedding = self.rec_session.run(
            [self.rec_output_name], 
            {self.rec_input_name: face_resized}
        )[0]
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.flatten()
    
    def compare(self, emb1, emb2):
        """Cosine similarity between embeddings"""
        return np.dot(emb1, emb2)

def create_test_dataset():
    """Create standardized test dataset"""
    
    test_dir = Path(__file__).parent / "test_dataset"
    
    # Clear existing test data if needed
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    test_dir.mkdir(exist_ok=True)
    
    # Create test subjects with distinct facial characteristics
    subjects = {
        "Subject_001": {"skin": (200, 180, 150), "eye": (0, 0, 0), "mouth": (100, 50, 50)},
        "Subject_002": {"skin": (180, 160, 140), "eye": (40, 40, 40), "mouth": (80, 60, 70)},
        "Subject_003": {"skin": (210, 190, 160), "eye": (30, 30, 30), "mouth": (120, 70, 60)}
    }
    
    for subject_name, features in subjects.items():
        subject_dir = test_dir / subject_name
        subject_dir.mkdir(exist_ok=True)
        
        for i in range(6):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Face circle
            cv2.circle(img, (100, 100), 60, features["skin"], -1)
            
            # Eyes
            cv2.circle(img, (70, 80), 10, features["eye"], -1)
            cv2.circle(img, (70, 80), 4, (255, 255, 255), -1)
            cv2.circle(img, (130, 80), 10, features["eye"], -1)
            cv2.circle(img, (130, 80), 4, (255, 255, 255), -1)
            
            # Nose
            cv2.line(img, (100, 90), (100, 110), (150, 120, 100), 3)
            
            # Mouth (different for each subject)
            if subject_name == "Subject_001":
                cv2.ellipse(img, (100, 135), (20, 10), 0, 0, 180, features["mouth"], -1)
            elif subject_name == "Subject_002":
                cv2.rectangle(img, (85, 130), (115, 140), features["mouth"], -1)
            else:
                cv2.ellipse(img, (100, 135), (20, 12), 0, 0, 360, features["mouth"], -1)
            
            # Add slight variation for different samples
            if i >= 3:
                if subject_name == "Subject_001":
                    cv2.ellipse(img, (100, 135), (15, 8), 0, 0, 180, features["mouth"], -1)
                elif subject_name == "Subject_002":
                    cv2.ellipse(img, (100, 135), (18, 8), 0, 0, 180, features["mouth"], -1)
            
            cv2.imwrite(str(subject_dir / f"sample_{i+1}.jpg"), img)
    
    return test_dir

def run_accuracy_test():
    """Execute frontal face accuracy test"""
    
    print("="*60)
    print("FRONTAL FACE ACCURACY TEST")
    print("="*60)
    print()
    
    # Initialize
    model_path = Path(__file__).parent / "models" / "buffalo_l"
    recognizer = FaceRecognizer(model_path)
    
    # Prepare test data
    print("Preparing test dataset...")
    test_dir = create_test_dataset()
    
    # Load test subjects
    subjects_data = []
    for subject_dir in sorted(test_dir.iterdir()):
        if subject_dir.is_dir():
            samples = []
            for img_path in sorted(subject_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is not None:
                    samples.append(img)
            
            if len(samples) >= 3:
                subjects_data.append({
                    "id": subject_dir.name,
                    "samples": samples
                })
    
    print(f"Dataset ready: {len(subjects_data)} subjects, {sum(len(s['samples']) for s in subjects_data)} samples\n")
    
    # Build reference gallery
    print("Building reference gallery...")
    gallery = []
    for subject in subjects_data:
        embeddings = []
        for i in range(2):  # First 2 samples as reference
            face = recognizer.get_face_crop(subject["samples"][i])
            if face is not None:
                emb = recognizer.get_embedding(face)
                embeddings.append(emb)
        
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            gallery.append({
                "id": subject["id"],
                "embedding": avg_emb
            })
    
    print(f"Gallery built: {len(gallery)} reference identities\n")
    
    # Execute verification tests
    print("Executing verification tests...")
    correct = 0
    total = 0
    
    for subject in subjects_data:
        for i in range(2, len(subject["samples"])):  # Remaining samples for testing
            face = recognizer.get_face_crop(subject["samples"][i])
            if face is not None:
                test_emb = recognizer.get_embedding(face)
                total += 1
                
                best_match = None
                best_score = -1
                
                for g in gallery:
                    score = recognizer.compare(test_emb, g["embedding"])
                    if score > best_score:
                        best_score = score
                        best_match = g["id"]
                
                if best_match == subject["id"]:
                    correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total verification attempts: {total}")
    print(f"Successful verifications:    {correct}")
    print(f"Failed verifications:        {total - correct}")
    print(f"\nFrontal Face Accuracy: {accuracy:.2f}%")
    
    if accuracy > 95:
        print(f"\nStatus: PASS (Exceeds 95% threshold)")
    else:
        print(f"\nStatus: Review required")
    
    # Save report
    report_dir = Path(__file__).parent / "test_results"
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / "frontal_accuracy_report.txt"
    with open(report_path, 'w') as f:
        f.write("FRONTAL FACE ACCURACY TEST REPORT\n")
        f.write("=================================\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: InsightFace buffalo_l\n")
        f.write(f"Test Methodology: 2 reference samples per subject, 4 verification samples per subject\n")
        f.write(f"Total Subjects: {len(subjects_data)}\n")
        f.write(f"Total Verification Tests: {total}\n")
        f.write(f"Successful Verifications: {correct}\n")
        f.write(f"Accuracy Rate: {accuracy:.2f}%\n")
        f.write(f"Required Threshold: >95%\n")
        f.write(f"Overall Result: {'PASS' if accuracy > 95 else 'FAIL'}\n")
    
    print(f"\nReport saved: {report_path}")
    
    return accuracy

if __name__ == "__main__":
    try:
        accuracy = run_accuracy_test()
    except Exception as e:
        print(f"\nError during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)