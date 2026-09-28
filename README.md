# Face Recognition with PCA (Eigenfaces)

This notebook builds a small **face recognition system** on top of **Principal Component Analysis (PCA)**, also known as the *eigenfaces* method.

**What it does**

1. Loads a dataset of 50×50 grayscale face images (10 people, 64 images each, split into 5 sets with increasingly difficult lighting conditions).
2. Pre-processes the images (histogram equalization, gamma correction, per-image standardization).
3. Learns a low-dimensional face space with PCA, using only the images of **Set 1** for training.
4. Measures how well faces from every set are recognized with a **nearest-neighbor classifier** in that space, for different numbers of principal components.
5. Compares PCA (eigen-decomposition of the covariance matrix) with the **SVD** of the data matrix.

**Contents**

| Task | Topic |
|------|-------|
| 2.1 | Data loading and pre-processing |
| 2.2 | PCA projection and reconstruction of a face |
| 2.3 | Reconstruction error vs. number of principal components |
| 2.4 | Visualizing the eigenfaces |
| 2.5 | Face recognition accuracy on every set |
| 2.6 | Reconstruction of faces from every set |
| 2.7 | Comparison with SVD |

**Requirements:** `numpy`, `matplotlib`, `opencv-python`, `scikit-learn`, and the dataset archive `faces.zip` placed next to this notebook (images named `personXX_YY.png`).
