import tensorflow as tf

IMG_SIZE = (224, 224)


def load_datasets(data_dir="data", batch_size=32, val_split=0.2, seed=42):
    """Folder layout: data/train/<scene_name>/*.jpg and data/test/<scene_name>/*.jpg"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train", validation_split=val_split, subset="training",
        seed=seed, image_size=IMG_SIZE, batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train", validation_split=val_split, subset="validation",
        seed=seed, image_size=IMG_SIZE, batch_size=batch_size,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test", image_size=IMG_SIZE, batch_size=batch_size, shuffle=False,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )
