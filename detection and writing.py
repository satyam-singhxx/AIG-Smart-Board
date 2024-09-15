import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from collections import deque

# Load the trained model and scaler
model = load_model('hand_gesture_model.h5')
scaler = joblib.load('scaler.pkl')

# Define the sequence length based on your model training
sequence_length = 30
input_shape = (sequence_length, 63)

# Initialize sequence buffer as a deque for better performance
sequence_buffer = deque(maxlen=sequence_length)

# Define gesture mapping
gesture_enum = pd.read_csv('data/gesture_enum.csv')
int_to_gesture = dict(zip(gesture_enum['enum'], gesture_enum['gesture']))

# Function to preprocess the frames
def preprocess_frame(frame):
    frame = np.array(frame).reshape(1, -1)
    frame = scaler.transform(frame)
    return frame.reshape(-1, 3)

# Function to predict gesture from the sequence
def predict_gesture(sequence):
    sequence = np.array(list(sequence))
    sequence = sequence.reshape(1, sequence_length, -1)
    prediction = model.predict(sequence, verbose=0)
    gesture_index = np.argmax(prediction)
    return int_to_gesture[gesture_index]

# Setting up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Color options
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
current_color = 0

# Initialize the MediaPipe Hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as hands:

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width, 3), np.uint8)

    writing = False
    erasing = False
    prev_index_tip = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)

        # Extract hand landmarks from the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess and add to sequence buffer
            processed_landmarks = preprocess_frame(keypoints)
            sequence_buffer.append(processed_landmarks)

            # Predict gesture if buffer is full
            if len(sequence_buffer) == sequence_length:
                gesture = predict_gesture(sequence_buffer)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture == 'write':
                    writing = True
                    erasing = False
                elif gesture == 'erase':
                    writing = False
                    erasing = True
                elif gesture == 'move':
                    writing = False
                    erasing = False

            # Get index finger tip coordinates
            index_tip = (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))

            # Perform writing or erasing
            if writing and prev_index_tip is not None:
                cv2.line(canvas, prev_index_tip, index_tip, colors[current_color], 2)
            elif erasing:
                # Calculate the bounding box of the palm
                x_coordinates = [int(lm.x * width) for lm in hand_landmarks.landmark]
                y_coordinates = [int(lm.y * height) for lm in hand_landmarks.landmark]
                x1, y1 = min(x_coordinates), min(y_coordinates)
                x2, y2 = max(x_coordinates), max(y_coordinates)
                
                # Erase using a filled rectangle
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)

            prev_index_tip = index_tip

            # Check for color selection
            if writing and hand_landmarks.landmark[8].y < 0.1:  # If index finger is at the top of the frame
                if hand_landmarks.landmark[8].x < 0.3:  # Left side for color selection
                    current_color = min(2, int(hand_landmarks.landmark[8].x * 10))

        # Draw color options
        for i, color in enumerate(colors):
            cv2.rectangle(frame, (10 + i*30, 10), (30 + i*30, 30), color, -1)
            if i == current_color:
                cv2.rectangle(frame, (10 + i*30, 10), (30 + i*30, 30), (255, 255, 255), 2)

        # Combine frame and canvas
        combined_image = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
        
        cv2.imshow('Air Writing with Gesture Recognition', combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()