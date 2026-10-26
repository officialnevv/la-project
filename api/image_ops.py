"""Image processing with NumPy matrices.
A colour image is a 3D array of shape (H, W, 3): one HxW matrix each for R, G, B.
A grayscale image is a single HxW matrix. Pixel values are 0..255."""
import base64, io
import numpy as np
from PIL import Image


def load_image(file, max_side=1200):
    """Read an image file -> uint8 array (H, W, 3)."""
    img = Image.open(file).convert("RGB")
    img.thumbnail((max_side, max_side))
    return np.asarray(img, dtype=np.uint8)


def to_grayscale(rgb):
    """Gray = 0.299 R + 0.587 G + 0.114 B  (matrix-vector product: (H,W,3) @ (3,))."""
    weights = np.array([0.299, 0.587, 0.114])
    return np.rint(rgb.astype(np.float64) @ weights).astype(np.uint8)


def adjust_brightness(img, beta):
    """I' = clip(I + beta, 0, 255). Adding beta to every entry (matrix + beta*ones)."""
    return np.clip(img.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)


def flip_horizontal(img):
    """Mirror left-right by reversing the column order."""
    return img[:, ::-1]


def flip_horizontal_matrix(img):
    """Same result via linear algebra: I' = I . J, where J is the exchange
    matrix (ones on the anti-diagonal). Used to prove the maths matches."""
    J = np.eye(img.shape[1])[::-1]
    out = np.einsum("hwc,wv->hvc", img.astype(np.float64), J) if img.ndim == 3 else img @ J
    return np.rint(out).astype(np.uint8)


def to_black_white(rgb, threshold=128):
    """B(i,j) = 255 if Gray(i,j) >= T else 0."""
    return np.where(to_grayscale(rgb) >= threshold, 255, 0).astype(np.uint8)


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(arr):
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(arr)).decode()


def apply_operation(rgb, op, brightness=0, threshold=128):
    if op == "grayscale":
        return to_grayscale(rgb)
    if op == "brightness":
        return adjust_brightness(rgb, brightness)
    if op == "flip":
        return flip_horizontal(rgb)
    if op == "bw":
        return to_black_white(rgb, threshold)
    raise ValueError(f"unknown operation: {op}")
