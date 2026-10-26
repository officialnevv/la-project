"""Local version: pick an image, apply all operations, show before/after, save results.
Usage: python main.py path/to/image.jpg [brightness] [threshold]"""
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
from api.image_ops import *

path = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")
beta = int(sys.argv[2]) if len(sys.argv) > 2 else 50
thr = int(sys.argv[3]) if len(sys.argv) > 3 else 128

rgb = load_image(path)
results = {
    "Original": rgb,
    "Grayscale": to_grayscale(rgb),
    f"Brightness {beta:+d}": adjust_brightness(rgb, beta),
    "Horizontal flip": flip_horizontal(rgb),
    f"Black & white (T={thr})": to_black_white(rgb, thr),
}
os.makedirs("output", exist_ok=True)
fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
for ax, (name, arr) in zip(axes, results.items()):
    ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(name); ax.axis("off")
    Image.fromarray(arr).save(f"output/{name.split(' (')[0].replace(' ', '_').replace('&', 'and')}.png")
plt.tight_layout(); plt.savefig("output/comparison.png"); plt.show()
