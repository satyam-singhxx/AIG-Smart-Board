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