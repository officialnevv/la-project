import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, request, jsonify
from image_ops import load_image, apply_operation, to_data_url

app = Flask(__name__)


@app.route("/api/process", methods=["POST"])
@app.route("/api/index", methods=["POST"])
def process():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded"), 400
    try:
        rgb = load_image(f)
        out = apply_operation(
            rgb,
            request.form.get("op", "grayscale"),
            brightness=int(request.form.get("brightness", 0)),
            threshold=int(request.form.get("threshold", 128)),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(original=to_data_url(rgb), result=to_data_url(out))
