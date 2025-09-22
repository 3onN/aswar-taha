# Fingerprint Enhancer — Colab Quickstart

This is the minimal package to run your fingerprint enhancement code on Google Colab.

## Files
- **theenhancer.py** : Your Python script containing the function `enhanced_upload_and_select()`.
- **requirements.txt** : Required Python packages for Colab or any Python 3.10+ environment.
- **README_Colab.md** : These instructions.

## How to Run on Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.

2. Install the required libraries (choose one):

   **If you uploaded requirements.txt**
   ```python
   !pip install -r requirements.txt
   ```

   **Or install directly**
   ```python
   !pip install opencv-python-headless numpy matplotlib Pillow scipy scikit-image
   ```

3. Upload your Python code:
   ```python
   from google.colab import files
   uploaded = files.upload()  # select theenhancer.py
   ```

4. Run your script:
   ```python
   %run theenhancer.py
   ```

5. Call the enhancement function (this will prompt you to upload the fingerprint image and apply the algorithm):
   ```python
   enhanced_upload_and_select()
   ```
