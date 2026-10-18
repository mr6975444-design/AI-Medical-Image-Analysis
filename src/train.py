import tensorflow as tf
import matplotlib.pyplot as plt

from src.data_loader import load_datasets
from src.model import build_model

from src.config import (
    EPOCHS,
    MODEL_PATH,
    PLOT_DIR,
)


def plot_training_history(history):
    """
    Save training and validation accuracy/loss graphs.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        PLOT_DIR / "class_weighted_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        PLOT_DIR / "class_weighted_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Training graphs saved.")


def train():

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_dataset, validation_dataset, _ = load_datasets()

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    normal_count = 1341
    pneumonia_count = 3875

    total = normal_count + pneumonia_count

    class_weight = {
        0: total / (2 * normal_count),
        1: total / (2 * pneumonia_count)
    }

    print("\nClass weights:")
    print(
        f"NORMAL: {class_weight[0]:.4f}"
    )
    print(
        f"PNEUMONIA: {class_weight[1]:.4f}"
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05
            ),
        ],
        name="data_augmentation"
    )

    train_dataset = train_dataset.map(
        lambda x, y: (
            augmentation(x, training=True),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING CLASS-WEIGHTED TRAINING")
    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    plot_training_history(history)

    print("\n")
    print("=" * 60)
    print("CLASS-WEIGHTED TRAINING COMPLETE")
    print("=" * 60)

    print("Model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()