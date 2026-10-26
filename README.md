# Image Processing Using Matrices

Linear Algebra project. Images are treated as matrices and processed with **Python + NumPy**.
A colour image is an `H × W × 3` array (R, G, B matrices, values 0–255). A grayscale image is one `H × W` matrix.

| Operation | Formula | NumPy |
|---|---|---|
| Grayscale | `G = 0.299R + 0.587G + 0.114B` (matrix–vector product) | `rgb @ [0.299, 0.587, 0.114]` |
| Brightness | `I' = clip(I + β, 0, 255)` | `np.clip(img + beta, 0, 255)` |
| Horizontal flip | `I' = I · J`, J = exchange matrix (ones on the anti-diagonal) | `img[:, ::-1]` |
| Black & white | `B(i,j) = 255 if G(i,j) ≥ T else 0` | `np.where(gray >= T, 255, 0)` |

## Run locally
```
pip install -r requirements.txt -r requirements-local.txt
python main.py path/to/image.jpg 50 128   # brightness β=50, threshold T=128
```
Results are saved in `output/`.

## Web app (Vercel)
`public/index.html` is the UI, `api/index.py` (Flask) is the backend, `api/image_ops.py` holds the matrix operations.
Deploy: import the GitHub repo at vercel.com/new (no settings needed), or `vercel --prod`.
Note: Vercel limits request bodies to ~4.5 MB, so upload photos under that size.
