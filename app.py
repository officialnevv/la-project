import base64
import io
import os
import sys

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Processing Using Matrices</title>
<style>
:root{--bg:#fafafa;--fg:#1a1a1a;--card:#fff;--bd:#ddd;--ac:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--card:#1e1e1e;--bd:#333;--ac:#60a5fa}}
body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1000px;margin:auto;padding:24px}
.box{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
img{max-width:100%;border-radius:6px}
label{display:block;margin:10px 0 4px}
select,input[type=file]{width:100%;padding:6px}
a.btn{display:inline-block;margin-top:10px;padding:8px 14px;background:var(--ac);color:#fff;border-radius:6px;text-decoration:none}
#err{color:#dc2626}
</style></head><body><main>
<h1>Image Processing Using Matrices</h1>
<p>Linear Algebra project: an image is a matrix of numbers, and each operation is a matrix operation.</p>
<div class="box">
  <input type="file" id="file" accept="image/*">
  <label>Operation</label>
  <select id="op">
    <option value="grayscale">Grayscale</option>
    <option value="brightness">Brightness</option>
    <option value="flip">Horizontal flip</option>
    <option value="bw">Black and white</option>
  </select>
  <div id="bBox" hidden><label>Brightness: <b id="bv">50</b></label><input type="range" id="brightness" min="-100" max="100" value="50" style="width:100%"></div>
  <div id="tBox" hidden><label>Threshold: <b id="tv">128</b></label><input type="range" id="threshold" min="0" max="255" value="128" style="width:100%"></div>
  <p id="err"></p>
</div>
<div class="grid">
  <div class="box"><h3>Original</h3><img id="orig" alt=""></div>
  <div class="box"><h3>Result</h3><img id="res" alt=""><br><a class="btn" id="dl" download="result.png" hidden>Save result</a></div>
</div>
<script>
const $ = id => document.getElementById(id);
let timer;

function updateControls() {
  const op = $("op").value;
  $("bBox").hidden = op !== "brightness";
  $("tBox").hidden = op !== "bw";
  $("bv").textContent = $("brightness").value;
  $("tv").textContent = $("threshold").value;
}

async function run() {
  updateControls();
  const file = $("file").files[0];
  if (!file) return;
  $("err").textContent = "";
  const data = new FormData();
  data.append("image", file);
  for (const k of ["op", "brightness", "threshold"]) data.append(k, $(k).value);
  try {
    const r = await fetch("/api/process", {method: "POST", body: data});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    $("orig").src = d.original;
    $("res").src = d.result;
    $("dl").href = d.result;
    $("dl").hidden = false;
  } catch (e) {
    $("err").textContent = e.message;
  }
}

["file", "op"].forEach(id => $(id).addEventListener("change", run));
["brightness", "threshold"].forEach(id => $(id).addEventListener("input", () => {
  updateControls();
  clearTimeout(timer);
  timer = setTimeout(run, 250);
}));
updateControls();
</script></main></body></html>"""


def load_image(file, max_side=1200):
    img = Image.open(file).convert("RGB")
    img.thumbnail((max_side, max_side))
    return np.asarray(img, dtype=np.uint8)


def to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114])
    return np.rint(rgb.astype(np.float64) @ weights).astype(np.uint8)


def adjust_brightness(img, beta):
    return np.clip(img.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)


def flip_horizontal(img):
    return img[:, ::-1]


def flip_horizontal_matrix(img):
    J = np.eye(img.shape[1])[::-1]
    if img.ndim == 3:
        out = np.einsum("hwc,wv->hvc", img.astype(np.float64), J)
    else:
        out = img @ J
    return np.rint(out).astype(np.uint8)


def to_black_white(rgb, threshold=128):
    return np.where(to_grayscale(rgb) >= threshold, 255, 0).astype(np.uint8)


def to_data_url(arr):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def apply_operation(rgb, op, brightness=0, threshold=128):
    if op == "grayscale":
        return to_grayscale(rgb)
    if op == "brightness":
        return adjust_brightness(rgb, brightness)
    if op == "flip":
        return flip_horizontal(rgb)
    if op == "bw":
        return to_black_white(rgb, threshold)
    raise ValueError("unknown operation: " + op)


@app.route("/")
def home():
    return PAGE


@app.route("/api/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded"), 400
    try:
        rgb = load_image(file)
        result = apply_operation(
            rgb,
            request.form.get("op", "grayscale"),
            int(request.form.get("brightness", 0)),
            int(request.form.get("threshold", 128)),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(original=to_data_url(rgb), result=to_data_url(result))


def run_local(path, beta=50, threshold=128):
    import matplotlib.pyplot as plt

    rgb = load_image(path)
    results = [
        ("original", rgb),
        ("grayscale", to_grayscale(rgb)),
        ("brightness", adjust_brightness(rgb, beta)),
        ("flip", flip_horizontal(rgb)),
        ("black_white", to_black_white(rgb, threshold)),
    ]

    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    for ax, (name, arr) in zip(axes, results):
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
        Image.fromarray(np.ascontiguousarray(arr)).save(os.path.join("output", name + ".png"))
    plt.tight_layout()
    plt.savefig(os.path.join("output", "comparison.png"))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        beta = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        run_local(sys.argv[1], beta, threshold)
    else:
        app.run(debug=True)
