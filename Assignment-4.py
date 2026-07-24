"""
Breast Cancer Classification using K-Nearest Neighbors (KNN)
----------------------------------------------------------------
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (Kaggle)
Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Download 'data.csv' from the Kaggle link above and place it in the same
folder as this script before running it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set(style="whitegrid")

TARGET_VARIABLE = "diagnosis"
K_NEIGHBORS = 5


# ---------------------------------------------------------------------
# Task 1: Data Understanding
# ---------------------------------------------------------------------
def load_and_explore_data(path="data.csv"):
    df = pd.read_csv(path)

    print("=" * 60)
    print("TASK 1: DATA UNDERSTANDING")
    print("=" * 60)
    print("\nShape of dataset:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe())

    numerical_features = [
        col for col in df.columns if col not in ["id", "diagnosis", "Unnamed: 32"]
    ]
    print("\nNumber of numerical features:", len(numerical_features))
    print("Target variable:", TARGET_VARIABLE)

    print("\nDiagnosis class distribution:")
    print(df["diagnosis"].value_counts())

    return df


# ---------------------------------------------------------------------
# Task 2: Data Preprocessing
# ---------------------------------------------------------------------
def preprocess_data(df):
    print("\n" + "=" * 60)
    print("TASK 2: DATA PREPROCESSING")
    print("=" * 60)

    print("\nMissing values per column (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    df_clean = df.copy()

    # Remove unnecessary columns
    cols_to_drop = [col for col in ["id", "Unnamed: 32"] if col in df_clean.columns]
    df_clean.drop(columns=cols_to_drop, inplace=True)
    print("\nDropped columns:", cols_to_drop)
    print("Missing values remaining after cleanup:", df_clean.isnull().sum().sum())

    # Encode target variable: M -> 1 (Malignant), B -> 0 (Benign)
    le = LabelEncoder()
    df_clean["diagnosis"] = le.fit_transform(df_clean["diagnosis"])
    print("Encoded classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    X = df_clean.drop("diagnosis", axis=1)
    y = df_clean["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize features - critical for KNN's distance calculations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining set size:", X_train_scaled.shape)
    print("Testing set size :", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------
# Task 3: Model Development
# ---------------------------------------------------------------------
def train_model(X_train_scaled, y_train):
    print("\n" + "=" * 60)
    print("TASK 3: MODEL DEVELOPMENT")
    print("=" * 60)

    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    knn.fit(X_train_scaled, y_train)

    print(f"\nTrained KNN classifier with K = {K_NEIGHBORS}")

    return knn


# ---------------------------------------------------------------------
# Task 4: Model Evaluation
# ---------------------------------------------------------------------
def evaluate_model(knn, X_train_scaled, X_test_scaled, y_train, y_test):
    print("\n" + "=" * 60)
    print("TASK 4: MODEL EVALUATION")
    print("=" * 60)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy Score: {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix - Breast Cancer Classification (KNN, K={K_NEIGHBORS})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    # Bonus: accuracy vs K plot
    k_values = range(1, 21)
    accuracies = []
    for k in k_values:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_scaled, y_train)
        accuracies.append(accuracy_score(y_test, knn_k.predict(X_test_scaled)))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker="o", color="navy")
    plt.axvline(x=K_NEIGHBORS, color="red", linestyle="--", label=f"K = {K_NEIGHBORS} (assignment value)")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Test Accuracy")
    plt.title("KNN Accuracy vs K")
    plt.xticks(list(k_values))
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png", dpi=150)
    plt.show()

    print("\nObservations:")
    print("1. With K=5 and standardized features, the model achieves strong accuracy,")
    print("   precision, recall, and F1-Score, showing good separation between classes.")
    print("2. False negatives (malignant predicted as benign) are the more dangerous")
    print("   error in this medical context, worth checking specifically in the matrix.")
    print("3. Accuracy is fairly stable across K values, with K=5 a reasonable middle")
    print("   ground between noise sensitivity (low K) and blurred boundaries (high K).")

    return y_pred, accuracy, precision, recall, f1


# ---------------------------------------------------------------------
# Task 5: Conclusion
# ---------------------------------------------------------------------
CONCLUSION = """
TASK 5: CONCLUSION
-------------------
This project used K-Nearest Neighbors (KNN) to classify breast tumors as
Malignant or Benign based on 30 diagnostic measurements derived from
digitized images of fine needle aspirates. After removing unnecessary
columns (id and the empty trailing column), encoding the target variable,
standardizing all feature values, and splitting the data 80/20, a KNN
classifier with K = 5 was trained and evaluated using Accuracy, Precision,
Recall, F1-Score, and a Confusion Matrix.

The key finding is that tumor measurements like radius, concavity, and
concave points are strong discriminators between malignant and benign
tumors, allowing KNN to achieve strong classification performance even with
a relatively simple, non-parametric algorithm.

Feature scaling is critical for KNN because the algorithm classifies a new
point based on the distance to its nearest neighbors; if features are left
on their original scales (e.g., area in the hundreds vs smoothness as a
small decimal), features with larger numeric ranges would dominate the
distance calculation and distort the results, regardless of their actual
importance.

A key limitation of KNN is that it is computationally expensive at
prediction time, since it must compute the distance from a new point to
every point in the training set - this doesn't scale well to very large
datasets, and KNN also has no explicit "model" to interpret afterward the
way Logistic Regression's coefficients can be interpreted.
"""


def main():
    df = load_and_explore_data("data.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    knn = train_model(X_train_scaled, y_train)
    evaluate_model(knn, X_train_scaled, X_test_scaled, y_train, y_test)
    print("\n" + "=" * 60)
    print(CONCLUSION)


if __name__ == "__main__":
    main()
