# Facial Emotion Recognition Using Deep CNN

## Project Overview

This project is a Deep Learning-based Facial Emotion Recognition system that detects human facial emotions using a Convolutional Neural Network (CNN).

The system uses grayscale facial images to classify expressions into four categories:

- Angry
- Happy
- Neutral
- Sad

It also supports real-time emotion detection through a webcam using OpenCV.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn

## Model Architecture

A custom Deep CNN architecture is implemented to extract facial features and classify emotions.

The architecture consists of:

1. Input Layer – 48 × 48 grayscale images
2. Convolutional Layers – Extract facial features
3. Max Pooling Layers – Reduce feature dimensions
4. Global Average Pooling – Summarize extracted features
5. Dense Layer – Learn emotion-related patterns
6. Dropout – Reduce overfitting
7. Softmax Output Layer – Classify four emotion categories

**Model Configuration:**
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 16
- Maximum Epochs: 30
- Early Stopping and Learning Rate Reduction

## Data Preprocessing

The project uses the following preprocessing and augmentation techniques:

- Grayscale image conversion
- Image resizing to 48 × 48 pixels
- Pixel normalization
- Random horizontal flipping
- Random rotation and zoom
- Random translation
- Brightness and contrast adjustments

Data augmentation helps improve model generalization.

## Real-Time Emotion Detection

The trained model is integrated with OpenCV for webcam-based emotion recognition.

Features include:

- Real-time face detection using Haar Cascade
- Facial emotion prediction
- Prediction confidence display
- Prediction smoothing using a 15-frame averaging window

**Keyboard Controls:**
- `Q` – Quit the webcam application
- `S` – Save a captured face image

## Project Structure

```text
Facial-Emotion-Recognition/
│
├── deep_cnn.py
├── webcam_emotion.py
├── emotion_deep_cnn.keras
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shwetha-Francis/Facial-Emotion-Recognition.git
```

### 2. Navigate to the Project

```bash
cd Facial-Emotion-Recognition
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Real-Time Emotion Detection

```bash
python webcam_emotion.py
```

### 5. Train the Model

If the dataset is available locally, run:

```bash
python deep_cnn.py
```

The training script creates the augmented training and validation datasets and saves the trained model.

## Evaluation

The training script evaluates the model using:

- Training Accuracy
- Validation Accuracy
- Training and Validation Loss
- Confusion Matrix
- Classification Report
- Precision, Recall and F1-Score

## Future Improvements

- Improve emotion classification accuracy.
- Experiment with different CNN architectures.
- Add additional emotion categories.
- Develop a user-friendly web interface using Streamlit.
- Improve face detection under different lighting conditions.

## Author

**Shwetha Francis**

B.Tech – Artificial Intelligence and Data Science

[GitHub Profile](https://github.com/Shwetha-Francis)
