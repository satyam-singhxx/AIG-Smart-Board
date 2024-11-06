import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from collections import deque
import time

# Load the trained model and scaler
model = load_model('hand_gesture_model.h5')
scaler = joblib.load('scaler.pkl')

# Reduced sequence length for faster prediction
sequence_length = 10  # Reduced from 30
input_shape = (sequence_length, 63)

# Initialize sequence buffer
sequence_buffer = deque(maxlen=sequence_length)

# Define gesture mapping
gesture_enum = pd.read_csv('data/gesture_enum.csv')
int_to_gesture = dict(zip(gesture_enum['enum'], gesture_enum['gesture']))

# Add gesture smoothing
class GestureSmoothing:
    def __init__(self, buffer_size=5):
        self.gesture_buffer = deque(maxlen=buffer_size)
        self.current_gesture = None
        
    def update(self, new_gesture):
        self.gesture_buffer.append(new_gesture)
        if len(self.gesture_buffer) == self.gesture_buffer.maxlen:
            most_common = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
            if self.gesture_buffer.count(most_common) >= 3:
                if most_common != self.current_gesture:
                    self.current_gesture = most_common
                    return True, most_common
        return False, self.current_gesture

def preprocess_frame(frame):
    return scaler.transform(frame.reshape(1, -1)).flatten()

def predict_gesture(sequence):
    if len(sequence) < sequence_length:
        return None
    sequence = np.array(list(sequence))
    sequence = sequence.reshape(1, sequence_length, -1)
    prediction = model.predict(sequence, verbose=0)
    return int_to_gesture[np.argmax(prediction)]

# Function to create blank canvas
def create_canvas(height, width, color='black'):
    if color == 'white':
        return np.full((height, width, 3), 255, np.uint8)
    return np.zeros((height, width, 3), np.uint8)

# Function to draw slider
def draw_slider(frame, x, y, width, value, min_val, max_val, label):
    cv2.rectangle(frame, (x, y), (x + width, y + 10), (150, 150, 150), -1)
    pos = int(x + (value - min_val) * width / (max_val - min_val))
    cv2.rectangle(frame, (pos - 5, y - 5), (pos + 5, y + 15), (200, 200, 200), -1)
    cv2.rectangle(frame, (pos - 5, y - 5), (pos + 5, y + 15), (100, 100, 100), 1)
    cv2.putText(frame, f"{label}: {value:.1f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return pos

# Setting up MediaPipe with optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Color options (Green, Blue, Red, White)
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 255)]
current_color = 0

# Initialize parameters
drawing_thickness = 2
canvas_opacity = 0.5
adjusting_thickness = False
adjusting_opacity = False
gesture_smoother = GestureSmoothing()

# Performance optimization settings
PREDICTION_INTERVAL = 0.1  # Seconds between predictions
last_prediction_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0) as hands:

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    
    canvas = create_canvas(height, width, 'black')
    canvas_color = 'black'

    writing = False
    erasing = False
    prev_index_tip = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Draw sliders
        thickness_pos = draw_slider(frame, 100, height - 100, 200, drawing_thickness, 1, 20, "Thickness")
        opacity_pos = draw_slider(frame, 100, height - 60, 200, canvas_opacity, 0, 1, "Opacity")

        current_time = time.time()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Optimize landmark drawing
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            index_tip = (int(hand_landmarks.landmark[8].x * width), 
                        int(hand_landmarks.landmark[8].y * height))

            # Handle slider adjustments
            if index_tip[1] > height - 120 and index_tip[1] < height - 80:
                if 100 <= index_tip[0] <= 300:
                    drawing_thickness = int(1 + (index_tip[0] - 100) * 19 / 200)
                    adjusting_thickness = True
                    writing = False
            elif index_tip[1] > height - 80 and index_tip[1] < height - 40:
                if 100 <= index_tip[0] <= 300:
                    canvas_opacity = (index_tip[0] - 100) / 200
                    adjusting_opacity = True
                    writing = False
            else:
                adjusting_thickness = False
                adjusting_opacity = False

                # Update sequence and predict at intervals
                processed_landmarks = preprocess_frame(keypoints)
                sequence_buffer.append(processed_landmarks)
                
                if current_time - last_prediction_time >= PREDICTION_INTERVAL:
                    gesture = predict_gesture(sequence_buffer)
                    if gesture:
                        changed, current_gesture = gesture_smoother.update(gesture)
                        if changed:
                            if current_gesture == 'write':
                                writing = True
                                erasing = False
                            elif current_gesture == 'erase':
                                writing = False
                                erasing = True
                            elif current_gesture == 'move':
                                writing = False
                                erasing = False
                    last_prediction_time = current_time

                # Color selection logic
                if writing and index_tip[1] < 50:
                    color_width = 30
                    color_spacing = 40
                    for i in range(len(colors)):
                        color_x_start = 10 + i * color_spacing
                        color_x_end = color_x_start + color_width
                        if color_x_start <= index_tip[0] <= color_x_end:
                            current_color = i
                            break

                # Handle drawing and erasing
                if writing and prev_index_tip is not None:
                    cv2.line(canvas, prev_index_tip, index_tip, colors[current_color], drawing_thickness)
                elif erasing:
                    x_coordinates = [int(lm.x * width) for lm in hand_landmarks.landmark]
                    y_coordinates = [int(lm.y * height) for lm in hand_landmarks.landmark]
                    x1, y1 = min(x_coordinates), min(y_coordinates)
                    x2, y2 = max(x_coordinates), max(y_coordinates)
                    erase_color = (255, 255, 255) if canvas_color == 'white' else (0, 0, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), erase_color, -1)

            prev_index_tip = index_tip

        # Draw color selection UI
        for i, color in enumerate(colors):
            cv2.rectangle(frame, (10 + i*40, 10), (40 + i*40, 40), color, -1)
            if i == current_color:
                cv2.rectangle(frame, (8 + i*40, 8), (42 + i*40, 42), (255, 255, 255), 2)

        # Add UI instructions
        cv2.putText(frame, "Press 'w': White canvas", (10, height-160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'b': Black canvas", (10, height-140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'c': Clear canvas", (10, height-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            canvas = create_canvas(height, width, 'white')
            canvas_color = 'white'
        elif key == ord('b'):
            canvas = create_canvas(height, width, 'black')
            canvas_color = 'black'
        elif key == ord('c'):
            canvas = create_canvas(height, width, canvas_color)
        elif key == ord('q'):
            break

        # Create final image
        combined_image = cv2.addWeighted(frame, 1, canvas, canvas_opacity, 0)
        
        # Display current gesture
        if gesture_smoother.current_gesture:
            cv2.putText(combined_image, f"Gesture: {gesture_smoother.current_gesture}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Air Writing with Gesture Recognition', combined_image)

    cap.release()
    cv2.destroyAllWindows()