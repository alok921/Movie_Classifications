"""Classify the scene in a movie clip every N seconds and save a timeline CSV."""
import argparse

import cv2
import numpy as np
import pandas as pd

from data import IMG_SIZE
from predict import load_assets


def main(args):
    model, classes = load_assets()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(fps * args.every)

    rows, frame_id = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, IMG_SIZE).astype("float32")
            probs = model.predict(np.expand_dims(rgb, 0), verbose=0)[0]
            i = int(np.argmax(probs))
            t = frame_id / fps
            rows.append({"time_sec": round(t, 1), "scene": classes[i], "confidence": float(probs[i])})
            print(f"{t:7.1f}s  {classes[i]:<12} {probs[i]:.2%}")
        frame_id += 1
    cap.release()

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved timeline to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=2.0, help="Seconds between sampled frames")
    ap.add_argument("--out", default="models/scene_timeline.csv")
    main(ap.parse_args())
