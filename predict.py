import argparse
import json

import numpy as np
import tensorflow as tf

from data import IMG_SIZE


def load_assets():
    model = tf.keras.models.load_model("models/scene_model.keras")
    with open("models/classes.json") as f:
        classes = json.load(f)
    return model, classes


def predict_image(path, model, classes, top=3):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top]
    return [(classes[i], float(probs[i])) for i in idx]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a movie frame / scene image")
    args = ap.parse_args()

    model, classes = load_assets()
    for name, p in predict_image(args.image, model, classes):
        print(f"{name}: {p:.2%}")
