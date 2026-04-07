 import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("flower_model.h5")

class_names = ['daisy','dandelion','rose','sunflower','tulip']


def decode_prediction(prediction: np.ndarray):
    scores = np.squeeze(prediction)
    scores = np.atleast_1d(scores).astype(float)

    if scores.size == 1:
        prob = float(np.clip(scores[0], 0.0, 1.0))
        idx = 1 if prob >= 0.5 else 0
        return class_names[idx]

    idx = int(np.argmax(scores))
    idx = min(idx, len(class_names) - 1)
    return class_names[idx]

def predict(img_path):
    img = image.load_img(img_path, target_size=(180,180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    return decode_prediction(prediction)