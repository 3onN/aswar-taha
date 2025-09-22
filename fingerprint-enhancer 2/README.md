# Fingerprint Enhancer (CLI + API)

Production-ready packaging of your Colab script for fingerprint enhancement.
Includes CLI, FastAPI API, Docker, CI, and environment config.
Your original Colab file is preserved at `app/legacy/theenhancer_colab.py`.

## Features
- Reuses your enhancement algorithms (Gabor/CLAHE/Ridge/etc.).
- Clean CLI for single/all methods.
- FastAPI endpoint to upload an image and choose a method.
- Dockerfile (+ `opencv-python-headless`).
- GitHub Actions CI.
- `.env` configuration.

## Quickstart (Local)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

### List Methods
```bash
python -m app.cli methods
```

### Enhance with a Specific Method
```bash
python -m app.cli enhance --input path/to/in.jpg --method "Gabor Filter" --output out.bmp
```

### Enhance with All Methods
```bash
python -m app.cli enhance_all --input path/to/in.jpg --out_dir outputs
```

## Run API (FastAPI)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

### API Endpoints
- `GET /health` → health check
- `GET /methods` → list available methods
- `POST /enhance` (multipart form: `method`, `file`) → returns enhanced PNG

## Docker
```bash
docker build -t fingerprint-enhancer:dev .
docker run --env-file .env -p 8000:8000 fingerprint-enhancer:dev
```

## Project Structure
```
app/
  legacy/theenhancer_colab.py  # Your original script
  enhancer.py                  # Clean wrapper
  cli.py                       # CLI commands
  main.py                      # FastAPI app
  config.py                    # Settings
tests/
.github/workflows/ci.yml
Dockerfile
requirements.txt
.env.example
README.md
```

## Notes
- We import your class and call its enhancement methods; no Colab upload/download is used in CLI/API.
- If a method list changes in your legacy file, `methods()` and `/methods` will reflect it at runtime.
- For servers, we use `opencv-python-headless` to avoid GUI deps.


## Colab Quickstart (Your Simple Flow)

If you prefer the simple Colab workflow you described, do this:

1. Open Google Colab → New Notebook.
2. Run the cells below (or use the provided notebook at `notebooks/Colab_Quickstart.ipynb`).

### Minimal cells:

```python
# 1) Install deps (quiet)
!pip -q install opencv-python-headless scikit-image scipy pillow matplotlib loguru

# 2) Upload your exact script (must define enhanced_upload_and_select)
from google.colab import files
print('Upload your theenhancer.py')
uploaded = files.upload()

# 3) Run the script, then call the function
%run theenhancer.py
enhanced_upload_and_select()  # this will prompt you to upload the fingerprint image and perform enhancement
```

This reproduces the same behavior: you upload the fingerprint image through the UI, and the function handles enhancement & display.
