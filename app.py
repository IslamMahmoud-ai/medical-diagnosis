"""
app.py — Flask REST API for Medical Diagnosis AI System
Endpoints:
  POST /predict      → diagnose a single image
  POST /predict/batch → diagnose multiple images
  GET  /health       → health check
"""

import os
import json
from flask import Flask, request, jsonify
import tensorflow as tf
from predict import predict, load_model

app   = Flask(__name__)
MODEL = None   # lazy-loaded on first request


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model("model/medical_model.h5")
    return MODEL


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "medical_diagnosis_v1"})


@app.route("/predict", methods=["POST"])
def diagnose():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmp_path = f"/tmp/{file.filename}"
    file.save(tmp_path)

    try:
        result = predict(tmp_path, get_model())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict/batch", methods=["POST"])
def diagnose_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    model   = get_model()
    for file in files:
        tmp = f"/tmp/{file.filename}"
        file.save(tmp)
        try:
            res = predict(tmp, model)
            res["file"] = file.filename
            results.append(res)
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
