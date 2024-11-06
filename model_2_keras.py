import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional
from tensorflow.keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class HandGestureRecognition:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
    
    def plot_training_history(self):
        """
        Plot training history including accuracy and loss curves
        """
        if self.history is None:
            raise ValueError("No training history available. Please train the model first.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plots saved as 'training_history.png'")

    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot confusion matrix for model predictions
        """
        if self.model is None:
            raise ValueError("No model available. Please train or load a model first.")
            
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix plot saved as 'confusion_matrix.png'")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes, 
                                 target_names=self.label_encoder.classes_))
    
    def preprocess_data(self, data_path):
        """
        Preprocess the data and prepare it for training
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Extract features and labels
        feature_cols = [col for col in data.columns if col not in ['gesture', 'sample_id', 'frame_id']]
        X = data[feature_cols]
        y = data['gesture']
        
        # Group by sample_id to ensure we get complete sequences
        grouped = data.groupby('sample_id')
        
        sequences = []
        labels = []
        
        for _, group in grouped:
            # Sort by frame_id within each group
            group = group.sort_values('frame_id')
            
            # Get features for this sequence
            seq_features = group[feature_cols].values
            
            # If the sequence is longer than sequence_length, we can create multiple sequences
            for i in range(0, len(seq_features) - self.sequence_length + 1, self.sequence_length):
                sequence = seq_features[i:i + self.sequence_length]
                if len(sequence) == self.sequence_length:
                    sequences.append(sequence)
                    # Use the gesture label from the first frame of the sequence
                    labels.append(group['gesture'].iloc[i])
        
        # Convert to numpy arrays
        X = np.array(sequences)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y_categorical = to_categorical(y_encoded)
        
        # Scale features
        original_shape = X.shape
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))
        X_reshaped = X_scaled.reshape(original_shape)
        
        print(f"Number of unique gestures: {len(self.label_encoder.classes_)}")
        print("Gesture classes:", self.label_encoder.classes_)
        print(f"Input shape: {X_reshaped.shape}")
        print(f"Output shape: {y_categorical.shape}")
        
        return X_reshaped, y_categorical

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            # First convolutional block
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),

            # Second convolutional block
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),

            # Third convolutional block
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),

            # Fourth convolutional block
            Conv1D(512, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),

            # Bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),

            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),

            # Dense layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_gesture_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

    def train(self, data_path, epochs=100, batch_size=32, validation_split=0.2):
        # Preprocess data
        X, y = self.preprocess_data(data_path)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Build the model
        input_shape = (self.sequence_length, X.shape[2])
        num_classes = y.shape[1]
        self.model = self.build_model(input_shape, num_classes)
        
        # Print the model summary
        self.model.summary()
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(X_val, y_val)
        
        # Calculate and print metrics
        test_loss, test_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal test accuracy: {test_accuracy:.4f}")
        print(f"Final test loss: {test_loss:.4f}")
        
        return self.history, test_accuracy

    def save_model(self, model_path='hand_gesture_model.keras', scaler_path='scaler_k.pkl', encoder_path='label_encoder.pkl'):
        """
        Save model, scaler, and label encoder
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
            
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_path='hand_gesture_model.keras', scaler_path='scaler.pkl', encoder_path='label_encoder.pkl'):
        """
        Load model, scaler, and label encoder
        """
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Model, scaler, and label encoder loaded successfully")
    
    def predict_gesture(self, sequence):
        """
        Predict gesture for a given sequence
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
            
        # Ensure sequence has correct features
        feature_cols = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
        if isinstance(sequence, pd.DataFrame):
            sequence = sequence[feature_cols].values
            
        # Preprocess the sequence
        scaled_sequence = self.scaler.transform(sequence)
        reshaped_sequence = scaled_sequence.reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction = self.model.predict(reshaped_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Convert back to gesture label
        predicted_gesture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_gesture, prediction[0][predicted_class]

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the gesture recognition system
        gesture_system = HandGestureRecognition(sequence_length=30)
        
        # Train the model
        history, accuracy = gesture_system.train(
            data_path='data/hand_historical.csv',  # Update with your data path
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Save the model
        gesture_system.save_model()
        
        print(f"\nTraining completed successfully!")
        print(f"Model and visualization files have been saved to the current directory.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise