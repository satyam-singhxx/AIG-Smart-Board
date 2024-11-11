# ✨ Air Writing with Gesture Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

> Transform your webcam into an interactive air writing canvas using hand gesture recognition! Write, erase, and create digital art with just your hand movements. 🖐️✍️

<p align="center">
  <img src="/api/placeholder/800/400" alt="Air Writing Demo">
</p>

## 🌟 Features

- ✍️ **Air Writing**: Draw in the air using hand gestures
- 🎨 **Multiple Colors**: Choose from green, blue, red, and white
- 🧹 **Gesture-Based Erasing**: Erase content with a simple hand gesture
- 🎯 **Real-time Recognition**: Instant response to hand movements
- 🎚️ **Adjustable Controls**: 
  - Drawing thickness
  - Canvas opacity
  - Canvas color (black/white)
- 💪 **Gesture Support**:
  - "Write" - Activate drawing mode
  - "Move" - Navigate without drawing
  - "Erase" - Clear selected area

## 🚀 Getting Started

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

Required packages:
- tensorflow
- opencv-python
- mediapipe
- numpy
- pandas
- scikit-learn
- joblib

### 🎯 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/satyam-singhxx/AIG-Smart-Board.git
cd AGI-smart-board
```

2. **Train the model** (optional - pre-trained model included)
```bash
python model_building.py
```

3. **Run the application**
```bash
python app.py
```

## 💡 Usage

1. **Data Collection**
```bash
python data_collection.py
```
- Follow on-screen instructions to record hand gestures
- Performs automatic data labeling and saving

2. **Model Training**
```bash
python model_1.py
```
- Trains the LSTM model on collected data
- Generates performance visualizations
- Saves the trained model

3. **Running the App**
```bash
python app_v2_h5.py
```

### 🎮 Controls

- **Gesture Controls**:
  - Write: Make a writing gesture
  - Move: Open palm
  - Erase: Closed fist

- **Keyboard Controls**:
  - `w`: Switch to white canvas
  - `b`: Switch to black canvas
  - `c`: Clear canvas
  - `q`: Quit application

- **UI Controls**:
  - Top color palette: Select drawing color
  - Bottom sliders: Adjust thickness and opacity

## 📊 Model Architecture

```
Model: Sequential
┌─────────────────────┬───────────┐
│ Layer (type)        │ Output    │
├─────────────────────┼───────────┤
│ LSTM               │ (None, 64) │
│ Dropout (0.2)      │           │
│ LSTM               │ (None, 32) │
│ Dropout (0.2)      │           │
│ Dense (softmax)    │ (None, 3)  │
└─────────────────────┴───────────┘
```

## 📁 Project Structure

```
air-writing-recognition/
├── data_collection.py      # Data collection script
├── model_building.py       # Model training script
├── app.py                 # Main application
├── data/                  # Collected data
│   ├── hand_keypoints.csv
│   ├── hand_historical.csv
│   └── gesture_enum.csv
├── models/                # Trained models
│   ├── hand_gesture_model.h5
│   └── scaler.pkl
└── visualizations/        # Training visualizations
    ├── training_history.png
    ├── confusion_matrix.png
    └── class_metrics.png
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- TensorFlow for deep learning capabilities
- OpenCV for image processing
- All contributors and supporters of the project

---

<p align="center">
  Made with ❤️ by [Your Name]
</p>
