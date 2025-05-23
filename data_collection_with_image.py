import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def preprocess_frame(frame):
    """
    Preprocesses a captured frame for keypoint extraction.
    Converts the frame to RGB color space.

    Args:
    - frame (np.array): Captured image from the camera.

    Returns:
    - preprocessed_image (np.array): Preprocessed image ready for keypoint extraction.
    """
    # Convert the BGR frame to RGB
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return preprocessed_image

def save_side_by_side(original, preprocessed, sample_id, gesture):
    """
    Saves a side-by-side comparison of the original and preprocessed images.

    Args:
    - original (np.array): Original captured image.
    - preprocessed (np.array): Preprocessed image.
    - sample_id (int): Sample identifier for saving files.
    - gesture (str): The gesture name for file naming.
    """
    # Resize preprocessed image to match original size
    preprocessed_resized = cv2.resize(preprocessed, (original.shape[1], original.shape[0]))

    # Concatenate the original and preprocessed images side by side
    side_by_side = np.hstack((original, preprocessed_resized))

    # Save the side-by-side image
    cv2.imwrite(f"data/images/{gesture}_comparison_{sample_id}.png", side_by_side)

def collect_hand_gesture_data(gestures, num_samples=100, sequence_length=30, delay_between_samples=3):
    cap = cv2.VideoCapture(0)

    # Create data directories
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Prepare CSV files
    keypoints_file = open('data/hand_keypoints.csv', 'w', newline='')
    keypoints_writer = csv.writer(keypoints_file)
    keypoints_writer.writerow(['gesture', 'sample_id'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']])
    
    historical_file = open('data/hand_historical.csv', 'w', newline='')
    historical_writer = csv.writer(historical_file)
    historical_header = ['gesture', 'sample_id', 'frame_id'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
    historical_writer.writerow(historical_header)
    
    gesture_enum_file = open('data/gesture_enum.csv', 'w', newline='')
    gesture_enum_writer = csv.writer(gesture_enum_file)
    gesture_enum_writer.writerow(['gesture', 'enum'])
    for i, gesture in enumerate(gestures):
        gesture_enum_writer.writerow([gesture, i])
    gesture_enum_file.close()

    keypoints_enum_file = open('data/hand_keypoints_enum.csv', 'w', newline='')
    keypoints_enum_writer = csv.writer(keypoints_enum_file)
    keypoints_enum_writer.writerow(['keypoint', 'enum'])
    for i in range(21):
        keypoints_enum_writer.writerow([f'HAND_LANDMARK_{i}', i])
    keypoints_enum_file.close()

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        for gesture in gestures:
            current_sample = 0
            
            while current_sample < num_samples:
                # Countdown timer
                for countdown in range(delay_between_samples, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        break

                    cv2.putText(frame, f"Next sample in: {countdown}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample: {current_sample}/{num_samples}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Prepare gesture...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Hand Gesture Collection', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):  # Wait for 1 second
                        break

                sequence_buffer = []
                frame_count = 0

                # Collect continuous data for the sequence
                while frame_count < sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        break

                    # Preprocess the captured frame
                    preprocessed_image = preprocess_frame(frame)
                    
                    results = hands.process(preprocessed_image)

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]  # Assuming we're tracking one hand
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        
                        sequence_buffer.append(keypoints)
                        frame_count += 1

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Display information on the camera window
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample: {current_sample}/{num_samples}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Collecting frames: {frame_count}/{sequence_length}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show the original frame
                    cv2.imshow('Hand Gesture Collection', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if len(sequence_buffer) == sequence_length:
                    # Save side-by-side comparison of original and preprocessed images
                    save_side_by_side(frame, preprocessed_image, current_sample, gesture)

                    # Save current frame keypoints (last frame of the sequence)
                    keypoints_row = [gesture, current_sample] + sequence_buffer[-1].tolist()
                    keypoints_writer.writerow(keypoints_row)
                    
                    # Save historical sequence
                    for frame_id, hist_keypoints in enumerate(sequence_buffer):
                        historical_row = [gesture, current_sample, frame_id] + hist_keypoints.tolist()
                        historical_writer.writerow(historical_row)

                    current_sample += 1
                
                if current_sample >= num_samples:
                    cv2.putText(frame, "Press 'n' for next gesture or 'q' to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Hand Gesture Collection', frame)
                    
                    key = cv2.waitKey(0) & 0xFF  # Wait for key press
                    if key == ord('q'):
                        break
                    elif key == ord('n'):
                        continue  # Move to the next gesture

            if key == ord('q'):
                break

    keypoints_file.close()
    historical_file.close()
    cap.release()
    cv2.destroyAllWindows()

# Example usage
gestures = ['write', 'move', 'erase']  # Add your hand gestures here
collect_hand_gesture_data(gestures, num_samples=2, sequence_length=30, delay_between_samples=1)
print("Data collection complete!")