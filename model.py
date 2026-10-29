import tensorflow as tf
from tensorflow.keras import layers

from data import IMG_SIZE


def build_model(num_classes):
    """MobileNetV2 (ImageNet weights) + new classification head.

    The model takes raw RGB pixels in the 0-255 range; preprocessing is inside the model.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = tf.keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)]
    )

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    compile_model(model, lr=1e-3)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
