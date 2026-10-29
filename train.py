import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data import load_datasets
from model import build_model, compile_model


def main(args):
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data, args.batch_size)
    print("Scene classes:", class_names)

    model, base = build_model(len(class_names))
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Stage 1: train only the new head
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early_stop])

    # Stage 2: fine-tune the last layers of the base network with a small learning rate
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, lr=1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[early_stop])

    # Evaluation on the test set
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=120)

    # Save model, labels and training curve
    model.save("models/scene_model.keras")
    with open("models/classes.json", "w") as f:
        json.dump(class_names, f)

    acc_hist = h1.history["accuracy"] + h2.history["accuracy"]
    val_hist = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.figure()
    plt.plot(acc_hist, label="train"); plt.plot(val_hist, label="val")
    plt.axvline(len(h1.history["accuracy"]) - 1, color="gray", linestyle="--", label="fine-tune start")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.savefig("models/accuracy_curve.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fine_tune_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
