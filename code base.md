code base


### Dependencies
import os
import subprocess
import sys

# Function to create a virtual environment
def create_virtual_env(env_name):
    # Check if venv module is available
    if sys.version_info >= (3, 3):
        subprocess.check_call([sys.executable, "-m", "venv", env_name])
        print(f"Virtual environment '{env_name}' created.")
    else:
        print("venv module not available for this version of Python.")

# Function to make the activate script executable on Unix/MacOS
def make_activate_executable(env_name):
    activate_path = f"./{env_name}/bin/activate"
    if os.name != 'nt':  # For Unix/MacOS
        if os.path.exists(activate_path):
            os.chmod(activate_path, 0o755)  # Give execute permissions
            print(f"'{activate_path}' made executable.")
        else:
            print(f"'{activate_path}' does not exist.")

# Function to install libraries
def install_libraries(env_name, libraries):
    # Activate the virtual environment
    if os.name == 'nt':  # For Windows
        activate_env = f".\\{env_name}\\Scripts\\activate"
    else:  # For Unix/MacOS
        activate_env = f"./{env_name}/bin/activate"

    # Install the required libraries
    subprocess.call(f"{activate_env} && pip install {' '.join(libraries)}", shell=True)

if __name__ == "__main__":
    env_name = "my_env"  # Change environment name if needed
    libraries = ["numpy", "pandas", "scikit-learn", "mediapipe", "opencv-python", "tensorflow", "matplotlib"]  # Specify the libraries to install

    # Step 1: Create a virtual environment in the current directory
    # create_virtual_env(env_name)

    # Step 2: Make activate script executable (Unix/MacOS)
    make_activate_executable(env_name)

    # Step 3: Install the required libraries
    install_libraries(env_name, libraries)


### Data collection

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def collect_hand_gesture_data(gestures, num_samples=100, sequence_length=30, delay_between_samples=3):
    cap = cv2.VideoCapture(0)
    
    # Create data directory
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

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

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
                    
                    cv2.imshow('Hand Gesture Collection', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if len(sequence_buffer) == sequence_length:
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
collect_hand_gesture_data(gestures, num_samples=10, sequence_length=30, delay_between_samples=1)
print("Data collection completed. Check the 'data' folder for CSV files.")  



### Model prepration
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data, sequence_length):
    X = []
    y = []
    
    for gesture in data['gesture'].unique():
        gesture_data = data[data['gesture'] == gesture]
        for sample_id in gesture_data['sample_id'].unique():
            sample = gesture_data[gesture_data['sample_id'] == sample_id]
            sample = sample.sort_values('frame_id')
            
            # Extract features (x, y, z coordinates for each landmark)
            features = sample.iloc[:, 3:].values
            
            # Ensure we have the correct sequence length
            if len(features) == sequence_length:
                X.append(features)
                y.append(gesture)
    
    return np.array(X), np.array(y)

# Load gesture enumeration
gesture_enum = pd.read_csv('data/gesture_enum.csv')
gesture_to_int = dict(zip(gesture_enum['gesture'], gesture_enum['enum']))

# Load and preprocess the data
data = load_data('data/hand_historical.csv')
X, y = preprocess_data(data, sequence_length=30)

# Convert gestures to integers
y = np.array([gesture_to_int[gesture] for gesture in y])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Convert labels to categorical
num_classes = len(gesture_to_int)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save('hand_gesture_model.h5')

# Save the scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")



### detection and writing
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