import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
model = load_model('hand_gesture_model.h5')
scaler = joblib.load('scaler.pkl')

# Define the sequence length based on your model training
sequence_length = 30
# Assuming 63 landmarks (x, y, z for 21 points)
input_shape = (sequence_length, 63)

# Initialize an empty sequence buffer
sequence_buffer = []

# Define gesture mapping (reverse of gesture_to_int)
gesture_enum = pd.read_csv('data/gesture_enum.csv')
int_to_gesture = dict(zip(gesture_enum['enum'], gesture_enum['gesture']))

# Function to preprocess the frames
def preprocess_frame(frame):
    frame = np.array(frame).reshape(1, -1)
    frame = scaler.transform(frame)
    return frame.reshape(-1, 3)

# Function to predict gesture from the sequence
def predict_gesture(sequence):
    sequence = np.array(sequence)
    sequence = sequence.reshape(1, sequence_length, -1)
    prediction = model.predict(sequence)
    gesture_index = np.argmax(prediction)
    return int_to_gesture[gesture_index]

# Setting up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the MediaPipe Hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6) as hands:

    # Open video capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    # Get the frame dimensions
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Create a blank canvas for drawing
    canvas = np.zeros((height, width, 3), np.uint8)

    # Variables for air writing
    writing = False
    erasing = False
    prev_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Extract hand landmarks from the frame
        def extract_landmarks(frame):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Assuming we're tracking one hand
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Return landmarks and the tip of the index finger
                return keypoints, (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))

            return None, None

        landmarks, finger_tip = extract_landmarks(frame)

        if landmarks is not None and landmarks.size > 0:
            # Preprocess the landmarks
            processed_landmarks = preprocess_frame(landmarks)

            # Add to sequence buffer
            sequence_buffer.append(processed_landmarks)

            # Keep the buffer size consistent
            if len(sequence_buffer) > sequence_length:
                sequence_buffer.pop(0)

            # If we have enough frames, predict gesture
            if len(sequence_buffer) == sequence_length:
                gesture = predict_gesture(sequence_buffer)
                print(f"Detected gesture: {gesture}")

                # Display the detected gesture on the video frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Handle writing and erasing based on gestures
                if gesture == 'write':
                    writing = True
                    erasing = False
                elif gesture == 'erase':
                    writing = False
                    erasing = True
                elif gesture == 'move':
                    writing = False
                    erasing = False

            # Perform writing or erasing
            if finger_tip:
                if writing and prev_point:
                    cv2.line(canvas, prev_point, finger_tip, (0, 255, 0), 2)
                elif erasing and prev_point:
                    cv2.line(canvas, prev_point, finger_tip, (0, 0, 0), 20)
                prev_point = finger_tip
            else:
                prev_point = None

        # Combine the frame and the canvas
        combined_image = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
        
        # Display the resulting frame
        cv2.imshow('Air Writing with Gesture Recognition', combined_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()