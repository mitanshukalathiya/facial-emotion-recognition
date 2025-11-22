import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "emotion_model.h5")
LABELS_FILE = os.path.join(MODEL_DIR, "emotion_labels.txt")


def load_model_and_labels():
    """Load the trained model and emotion labels."""
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_FILE)

    with open(LABELS_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"[INFO] Loaded classes: {labels}")
    return model, labels


def preprocess_face(gray_img, coords):
    """Crop, resize, and normalize the detected face region."""
    x, y, w, h = coords
    roi = gray_img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))
    return roi


def main():
    model, labels = load_model_and_labels()

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Cannot access camera.")
        return

    print("[INFO] Press 'q' to quit the window.")
    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Frame not captured.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = preprocess_face(gray, (x, y, w, h))
            predictions = model.predict(face, verbose=0)
            label = labels[np.argmax(predictions)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
