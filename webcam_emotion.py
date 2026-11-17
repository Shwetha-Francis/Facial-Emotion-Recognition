import cv2
import tensorflow as tf
import numpy as np
from collections import deque


# LOAD TRAINED CNN MODEL
model = tf.keras.models.load_model("emotion_deep_cnn.keras")

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# EMOTION NAMES — must match train_data.class_names order (alphabetical): ['Angry', 'happy', 'neutral']
classes = ["Angry","Happy","Neutral","sad"]


# LOAD FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# TEMPORAL SMOOTHING: average predictions over the last N frames
SMOOTHING_WINDOW = 15
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# CROP PADDING: extra margin around the Haar box, as a fraction of box size
PADDING_RATIO = 0.20

# Track last printed emotion so we only print on change
last_printed_emotion = None


# OPEN WEBCAM
cap = cv2.VideoCapture(0)

print("Starting webcam...")
print("Press S to save a debug snapshot of the detected face.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()
    if not ret:
        print("Could not access webcam.")
        break

    # Read the key once per frame (not once per detected face)
    key = cv2.waitKey(1) & 0xFF

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # PAD THE CROP so small jitter in the Haar box doesn't reframe the face every frame
        pad_w = int(w * PADDING_RATIO)
        pad_h = int(h * PADDING_RATIO)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(gray.shape[1], x + w + pad_w)
        y2 = min(gray.shape[0], y + h + pad_h)

        face = gray[y1:y2, x1:x2]

        # HISTOGRAM EQUALIZATION (normalizes lighting differences vs training set)
        face = cv2.equalizeHist(face)

        # RESIZE + NORMALIZE (match training preprocessing)
        face = gray[y1:y2, x1:x2]

        face_resized = cv2.resize(
         face,
         (48, 48),
         interpolation=cv2.INTER_AREA
        )

        face_input = face_resized.astype(np.float32) / 255.0

        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)

        raw_prediction = model.predict(
        face_input,
        verbose=0
        )[0]

        # TEMPORAL SMOOTHING: average this frame's prediction with recent frames
        prediction_history.append(raw_prediction)
        prediction = np.mean(prediction_history, axis=0)

        emotion_index = np.argmax(prediction)
        emotion = classes[emotion_index]
        confidence = prediction[emotion_index] * 100

        # PRINT ONLY WHEN THE PREDICTED EMOTION CHANGES
        if emotion != last_printed_emotion:
            print(f"Predicted: {emotion}")
            last_printed_emotion = emotion

        # PRESS S TO SAVE A CLEAN DEBUG SNAPSHOT
        if key == ord("s"):
            cv2.imwrite("debug_face.jpg", face_resized)
            print("Saved debug_face.jpg")

        # DRAW BOX + LABEL (padded box, matches what the model actually sees)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{emotion} {confidence:.1f}%"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection - Press Q to quit", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")