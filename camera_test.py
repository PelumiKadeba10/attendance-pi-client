import cv2
import sys

def main():
    # Initialize the USB camera (0 is typically the built-in or first USB camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera. Check your USB connection.")
        sys.exit()

    print("Successfully connected to the camera!")
    print("\n--- CONTROLS ---")
    print("Press 'W' to INCREASE brightness")
    print("Press 'S' to DECREASE brightness")
    print("Press 'Q' to QUIT")
    print("----------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Get the current brightness value from the camera driver
        current_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        
        # Overlay the current brightness value onto the video stream
        cv2.putText(frame, f"Brightness: {current_brightness:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the live feed
        cv2.imshow("Raspberry Pi Camera Test", frame)

        # Wait for a key press (10ms delay)
        key = cv2.waitKey(10) & 0xFF

        # 'q' key to exit
        if key == ord('q'):
            break
        
        # 'w' key to increase brightness
        elif key == ord('w'):
            # OpenCV brightness scales vary by camera driver (e.g., 0.0 to 1.0 OR 0 to 255)
            # Adjust the step size (+0.05 or +5) depending on your camera's scale
            new_brightness = current_brightness + 0.05 
            cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
            
        # 's' key to decrease brightness
        elif key == ord('s'):
            new_brightness = current_brightness - 0.05
            cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)

    # Clean up and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test ended successfully.")

if __name__ == "__main__":
    main()