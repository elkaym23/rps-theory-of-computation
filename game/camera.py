import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import random
import os

moves = ["rock", "paper", "scissors", "invalid"]

# Load model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../RPS/game
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model.savedmodelv3.0"))

print("Loading model from:", MODEL_DIR)

model = tf.keras.Sequential([
    TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
])


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    # Stream frames to Flask
    def get_frame(self):
        ret, frame = self.cap.read()
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    # Predict RPS from one frame
    def predict(self):
        ret, frame = self.cap.read()
        img = cv2.resize(frame, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        pred = model(img)
        prediction = list(pred.values())[0].numpy()[0]

        user_move = moves[np.argmax(prediction)]

        # If user made an invalid gesture, AI shouldn't pick a move.
        ai_move = random.choice(moves[:3]) if user_move != "invalid" else "none"

        return {
            "user": user_move,
            "ai": ai_move,
            "rock": float(prediction[0]),
            "paper": float(prediction[1]),
            "scissors": float(prediction[2]),
            "invalid": float(prediction[3])
        }