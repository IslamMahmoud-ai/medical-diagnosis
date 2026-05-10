"""
train.py — Medical Diagnosis AI System
Train ResNet50-based model for multi-class disease classification.
Usage: python train.py --data_dir ./data --epochs 20 --classes 5
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

CLASS_NAMES = ["normal", "pneumonia", "tumor", "fracture", "other"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32


# ─── Model ────────────────────────────────────────────────────────
def build_model(num_classes: int) -> Model:
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*IMG_SIZE, 3))
    base.trainable = False          # freeze pretrained weights

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base.input, out)


# ─── Data Generators ──────────────────────────────────────────────
def get_generators(data_dir: str):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_flow = train_gen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training"
    )
    val_flow = train_gen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation"
    )
    return train_flow, val_flow


# ─── Fine-tune ────────────────────────────────────────────────────
def unfreeze_and_finetune(model: Model, train_flow, val_flow,
                          epochs: int = 10):
    """Unfreeze top layers of ResNet50 for fine-tuning."""
    for layer in model.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_flow, validation_data=val_flow, epochs=epochs)


# ─── Train ────────────────────────────────────────────────────────
def train(data_dir: str, num_classes: int = 5, epochs: int = 20):
    model = build_model(num_classes)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_flow, val_flow = get_generators(data_dir)

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
        callbacks.ModelCheckpoint("model/best_model.h5",
                                  save_best_only=True),
    ]

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        callbacks=cb,
    )

    # Fine-tune phase
    unfreeze_and_finetune(model, train_flow, val_flow, epochs=10)

    os.makedirs("model", exist_ok=True)
    model.save("model/medical_model.h5")
    print("✅ Model saved → model/medical_model.h5")

    # Plot
    _plot_history(history)
    return model


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],   label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--classes",  type=int, default=5)
    args = parser.parse_args()
    train(args.data_dir, args.classes, args.epochs)
