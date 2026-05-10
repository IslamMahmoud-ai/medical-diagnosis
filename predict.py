"""
predict.py — Inference utilities for Medical Diagnosis AI System
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

CLASS_NAMES = ["normal", "pneumonia", "tumor", "fracture", "other"]
IMG_SIZE    = (224, 224)


def load_model(model_path: str = "model/medical_model.h5"):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img_path: str) -> np.ndarray:
    """Load, resize, normalize an image → (1, 224, 224, 3)."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict(img_path: str, model) -> dict:
    """Return diagnosis label + confidence + all class probabilities."""
    tensor = preprocess_image(img_path)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return {
        "diagnosis":   CLASS_NAMES[idx],
        "confidence":  round(float(probs[idx]) * 100, 2),
        "all_classes": {cls: round(float(p) * 100, 2)
                        for cls, p in zip(CLASS_NAMES, probs)},
    }


def predict_batch(img_dir: str, model) -> list[dict]:
    """Run inference on all images in a directory."""
    results = []
    for p in Path(img_dir).glob("*.jpg"):
        try:
            res = predict(str(p), model)
            res["file"] = p.name
            results.append(res)
        except Exception as e:
            results.append({"file": p.name, "error": str(e)})
    return results


if __name__ == "__main__":
    import sys, json
    model = load_model()
    result = predict(sys.argv[1], model)
    print(json.dumps(result, indent=2))
