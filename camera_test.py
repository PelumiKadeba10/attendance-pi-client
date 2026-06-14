import cv2
import time
import os

def test_brightness_levels():
    # Initialize the camera on the blue USB port
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Double-check the USB connection.")
        return

    # Define the brightness levels you want to test
    # Note: If your camera uses the 0-255 scale instead of 0.0-1.0, 
    # you can change these to values like 100, 180, and 240.
    levels = {
        "low_boost": 0.6,
        "medium_boost": 0.75,
        "high_boost": 0.9
    }

    print("Camera initialized. Starting brightness sweep...")

    for label, value in levels.items():
        print(f"Setting brightness to {value} ({label})...")
        cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
        
        # Verify if the hardware accepted the value
        actual_val = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        print(f"Driver reports brightness is: {actual_val}")

        # Let the sensor adjust to the new exposure setting
        time.sleep(1.5)
        
        # Flush the buffer to get a fresh frame
        for _ in range(5):
            cap.read()
            
        ret, frame = cap.read()
        if ret:
            filename = f"test_{label}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        else:
            print(f"Failed to capture frame at brightness {value}")

    cap.release()
    print("\nSweep complete! Check your folder for the new images.")

if __name__ == "__main__":
    test_brightness_levels()