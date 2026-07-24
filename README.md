# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
Build a K-Nearest Neighbors (KNN) classification model that predicts
whether a breast tumor is **Malignant (M)** or **Benign (B)** based on
diagnostic measurements from digitized images of a fine needle aspirate
(FNA) of a breast mass.

## Dataset Link
Breast Cancer Wisconsin Diagnostic Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> The dataset is **not** included in this repository. Download `data.csv`
> from the Kaggle link above and place it in the project's root folder
> before running the notebook/script.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization
- `scikit-learn` — train/test split, standardization, KNN classifier, and
  evaluation metrics

Install everything with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Methodology
1. **Data Understanding** — Loaded the dataset with Pandas, inspected the
   first five records, dataset info, and summary statistics. Identified:
   - Numerical features: 30 measurements (mean, standard error, and worst
     values of radius, texture, perimeter, area, smoothness, compactness,
     concavity, concave points, symmetry, and fractal dimension)
   - Target variable: `diagnosis` (M = Malignant, B = Benign)
2. **Data Preprocessing**
   - Checked for missing values.
   - Removed unnecessary columns: `id` (identifier) and `Unnamed: 32` (an
     empty trailing column present in the raw CSV, if applicable).
   - Encoded the target variable (`M` → 1, `B` → 0) with Label Encoding.
   - Standardized all feature values with `StandardScaler` — essential for
     KNN since it relies on distance calculations.
   - Split the data into 80% training and 20% testing sets (stratified on
     `diagnosis` to preserve class balance).
3. **Model Development**
   - Trained a `KNeighborsClassifier` with **K = 5**.
   - Predicted class labels for the test set.
4. **Model Evaluation**
   - Computed Accuracy, Precision, Recall, and F1-Score.
   - Generated a Confusion Matrix (visualized as a heatmap).
   - (Bonus) Plotted accuracy vs. K for K = 1 to 20, to visually justify why
     K = 5 is a reasonable choice.

## Results
- The KNN model (K = 5) achieves high accuracy, precision, recall, and
  F1-Score on the test set, indicating that the diagnostic measurements
  separate malignant and benign tumors well in feature space.
- The confusion matrix shows very few misclassifications; false negatives
  (malignant tumors predicted as benign) are the more clinically important
  error type to monitor, even if the overall accuracy is high.
- The accuracy-vs-K plot shows performance is fairly stable across a range
  of K values, with very low K being more sensitive to noise and very high
  K blurring the decision boundary — K = 5 sits in a reasonable middle
  ground.
- Exact Accuracy / Precision / Recall / F1-Score values are printed when
  the notebook/script is run against `data.csv` (values will vary slightly
  depending on the train/test split).

## Conclusion
This project used K-Nearest Neighbors to classify breast tumors as
Malignant or Benign based on 30 diagnostic measurements. Tumor measurements
like radius, concavity, and concave points proved to be strong
discriminators between the two classes, letting a relatively simple,
non-parametric algorithm achieve strong classification performance.
**Feature scaling is critical for KNN** because it classifies based on
distance to neighboring points — leaving features on their original scales
would let large-magnitude features (like `area`) dominate the distance
calculation regardless of their real importance. A key **limitation of
KNN** is that it is computationally expensive at prediction time, since it
must compute distances to every training point, and it offers no
interpretable model coefficients the way Logistic Regression does — making
it harder to explain *why* a particular prediction was made, which matters
in a healthcare context.

## Repository Structure
```
.
├── Assignment-4.ipynb   # Jupyter notebook version (recommended)
├── Assignment-4.py      # Plain Python script version
└── README.md
```

## How to Run
```bash
# 1. Place data.csv (downloaded from Kaggle) in this folder
# 2. Run the notebook
jupyter notebook Assignment-4.ipynb

# OR run the script directly
python Assignment-4.py
```
