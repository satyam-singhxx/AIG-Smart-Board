import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set style for better visualizations
plt.style.use('default')

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
            
            features = sample.iloc[:, 3:].values
            
            if len(features) == sequence_length:
                X.append(features)
                y.append(gesture)
    
    return np.array(X), np.array(y)

# Function to plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Function to plot per-class metrics
def plot_class_metrics(report_dict, classes):
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = pd.DataFrame({
        metric: [report_dict[cls][metric] for cls in classes]
        for metric in metrics
    }, index=classes)
    
    plt.figure(figsize=(12, 6))
    class_metrics.plot(kind='bar')
    plt.title('Per-class Performance Metrics')
    plt.xlabel('Gesture Class')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_metrics.png')
    plt.close()

# Main execution
# Load gesture enumeration
gesture_enum = pd.read_csv('data/gesture_enum.csv')
gesture_to_int = dict(zip(gesture_enum['gesture'], gesture_enum['enum']))
int_to_gesture = dict(zip(gesture_enum['enum'], gesture_enum['gesture']))

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

# Store original test labels for later use
y_test_original = y_test.copy()

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

# Print model summary
print("Model Architecture:")
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, 
                   validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = y_test_original

# Plot training history
plot_training_history(history)

# Plot confusion matrix
plot_confusion_matrix(y_test_classes, y_pred_classes, 
                     [int_to_gesture[i] for i in range(num_classes)])

# Generate and plot classification report
report = classification_report(y_test_classes, y_pred_classes, 
                             target_names=[int_to_gesture[i] for i in range(num_classes)],
                             output_dict=True)
plot_class_metrics(report, [int_to_gesture[i] for i in range(num_classes)])

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, 
                          target_names=[int_to_gesture[i] for i in range(num_classes)]))

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Save the model and scaler
model.save('hand_gesture_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("\nAll visualizations, model, and training history have been saved successfully.")
print("\nFiles generated:")
print("1. training_history.png - Training and validation accuracy/loss curves")
print("2. confusion_matrix.png - Confusion matrix heatmap")
print("3. class_metrics.png - Per-class performance metrics")
print("4. training_history.csv - Detailed training history")
print("5. hand_gesture_model.h5 - Trained model")
print("6. scaler.pkl - Fitted StandardScaler")