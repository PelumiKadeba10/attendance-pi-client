import cv2
import time

def test_headless():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera connected successfully over SSH!")
    
    # Get default brightness
    print(f"Default brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    
    # Try to boost brightness (adjust value based on your camera)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
    print(f"New brightness set to: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    
    # Let the camera warm up for a second
    time.sleep(1)
    
    # Capture a single frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_image.jpg", frame)
        print("Saved a test snapshot to 'test_image.jpg'. Check this file to see if it's bright enough!")
    else:
        print("Failed to capture image.")

    cap.release()

if __name__ == "__main__":
    test_headless()