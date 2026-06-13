#!/usr/bin/env python3
"""
Download sample face images for testing
Uses free, open-source face images from IMDB-WIKI sample
"""

import urllib.request
import os
from pathlib import Path
import zipfile
import cv2
import numpy as np

def download_test_images():
    """Download a small set of test face images"""
    
    print("="*60)
    print("DOWNLOADING TEST FACE IMAGES")
    print("="*60)
    
    # Create test_photos directory
    test_dir = Path(__file__).parent / "test_photos"
    test_dir.mkdir(exist_ok=True)
    
    # Alternative: Use pre-existing sample images from sklearn
    print("\nOption: Using built-in sample images from sklearn...")
    
    try:
        from sklearn.datasets import fetch_lfw_people
        
        print("Loading LFW mini dataset (small version)...")
        # Load just 5 people with fewer images (much smaller download)
        lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
        
        print(f"Loaded {len(lfw.images)} images of {len(lfw.target_names)} people")
        
        # Organize by person
        for person_id, person_name in enumerate(lfw.target_names):
            person_dir = test_dir / person_name.replace(' ', '_')
            person_dir.mkdir(exist_ok=True)
            
            # Get indices for this person
            indices = np.where(lfw.target == person_id)[0]
            
            # Save up to 5 images per person
            for idx, img_idx in enumerate(indices[:5]):
                img = lfw.images[img_idx]
                # Convert to BGR
                img_uint8 = (img * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                
                # Save
                img_path = person_dir / f"{idx+1}.jpg"
                cv2.imwrite(str(img_path), img_bgr)
                print(f"  Saved: {person_name} - image {idx+1}")
            
            if len(indices) >= 3:
                print(f"✓ {person_name}: {min(5, len(indices))} images saved")
        
        print(f"\n✓ Test images saved to {test_dir}/")
        return True
        
    except Exception as e:
        print(f"Error with LFW: {e}")
        return False

def create_synthetic_test():
    """Create synthetic test faces as fallback"""
    
    print("\nCreating synthetic test faces...")
    
    test_dir = Path(__file__).parent / "test_photos"
    test_dir.mkdir(exist_ok=True)
    
    # Create 3 synthetic "people" with variations
    persons = ['Alice', 'Bob', 'Charlie']
    
    for person in persons:
        person_dir = test_dir / person
        person_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            # Create a simple synthetic face-like pattern
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Face circle
            cv2.circle(img, (100, 100), 60, (200, 180, 150), -1)
            
            # Eyes
            cv2.circle(img, (70, 80), 10, (0, 0, 0), -1)
            cv2.circle(img, (130, 80), 10, (0, 0, 0), -1)
            cv2.circle(img, (70, 80), 5, (255, 255, 255), -1)
            cv2.circle(img, (130, 80), 5, (255, 255, 255), -1)
            
            # Mouth (different for each person)
            if person == 'Alice':
                cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (100, 50, 50), -1)
            elif person == 'Bob':
                cv2.rectangle(img, (85, 125), (115, 135), (80, 60, 70), -1)
            else:  # Charlie
                cv2.ellipse(img, (100, 130), (20, 15), 0, 0, 360, (120, 70, 60), -1)
            
            # Add some variation
            if i >= 3:  # Different expressions
                cv2.circle(img, (100, 100), 60, (210, 190, 160), -1)
            
            cv2.imwrite(str(person_dir / f"{i+1}.jpg"), img)
        
        print(f"✓ Created synthetic images for {person}")
    
    return True

def verify_test_images():
    """Verify test images are ready"""
    
    test_dir = Path(__file__).parent / "test_photos"
    
    if not test_dir.exists():
        return False
    
    persons = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if len(persons) < 2:
        return False
    
    print("\n" + "="*60)
    print("TEST IMAGES READY")
    print("="*60)
    
    for person_dir in persons:
        images = list(person_dir.glob("*.jpg"))
        print(f"  {person_dir.name}: {len(images)} images")
        
        if len(images) < 3:
            print(f"    ⚠ Warning: Need at least 3 images for testing")
    
    return len(persons) >= 2

if __name__ == "__main__":
    # Try to download real face images
    print("Attempting to download test face images...")
    success = download_test_images()
    
    if not success:
        print("\nFalling back to synthetic test images...")
        create_synthetic_test()
    
    verify_test_images()
    
    print("\n✓ Setup complete! Now run: python test_with_own_photos.py")