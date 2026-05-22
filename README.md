# Iris Flower Classification

A machine learning pipeline that trains and evaluates multiple classifiers on the Iris dataset, comparing performance across models using scikit-learn.

## Overview

This project builds a full end-to-end classification pipeline starting from raw data exploration all the way to trained models with detailed evaluation reports. The pipeline runs every model twice: once on all engineered features, and once after applying feature selection, so we can measure the impact of trimming the feature space.

**Models compared:**
- Logistic Regression
- Ridge Classifier
- Random Forest (100 trees, max depth 2, Gini criterion)
- Gradient Boosting (100 estimators, max depth 2)

**Evaluation metrics:** Accuracy, F1 Score (macro), R²

## Project Structure

```
Iris_Flower_Classification/
├── Iris_Flower_Classification.ipynb   # Main notebook
├── Data Visualization/
│   ├── Target Distribution.png
│   ├── Box Plot (MAIN).png
│   ├── Box Plot (Feature Extracted).png
│   ├── Correlation Matrix (MAIN).png
│   ├── Correlation Matrix (Feature Extracted).png
│   └── Correlation Matrix (Feature Selected).png
└── Results/
    ├── No Feature Selection/
    │   ├── LogisticRegression Model Evaluation.png
    │   ├── LogisticRegression Confusion Matrix.png
    │   ├── RidgeClassifier Model Evaluation.png
    │   ├── RidgeClassifier Confusion Matrix.png
    │   ├── RandomForestClassifier Model Evaluation.png
    │   ├── RandomForestClassifier Confusion Matrix.png
    │   ├── GradientBoostingClassifier Model Evaluation.png
    │   └── GradientBoostingClassifier Confusion Matrix.png
    └── With Feature Selection/
        ├── LogisticRegression Model Evaluation.png
        ├── LogisticRegression Confusion Matrix.png
        ├── RidgeClassifier Model Evaluation.png
        ├── RidgeClassifier Confusion Matrix.png
        ├── RandomForestClassifier Model Evaluation.png
        ├── RandomForestClassifier Confusion Matrix.png
        ├── GradientBoostingClassifier Model Evaluation.png
        └── GradientBoostingClassifier Confusion Matrix.png
```

## Pipeline Walkthrough

### 1. Data Loading

The Iris dataset is loaded directly from `sklearn.datasets`. It contains 150 samples across three flower species   Setosa, Versicolor, and Virginica   with four measurements per sample: sepal length, sepal width, petal length, and petal width. The dataset is perfectly balanced with 50 samples per class.

### 2. Exploratory Data Analysis

Before touching the models, we spend time understanding the data. We look at the statistical summary, confirm there are no missing values, check the class distribution, and visualise the correlations between features. The key finding here is that petal length and petal width are very strongly correlated with the target (~0.95), making them the most informative features for classification.

### 3. Feature Engineering

The original four measurements are extended into 14 features by computing geometric properties for both the sepal and petal:

| Feature | Formula |
|---|---|
| Area | length × width |
| Ratio | length / width |
| Diagonal | √(length² + width²) |
| Mean | (length + width) / 2 |
| Difference | length − width |

These derived features capture shape information that the raw measurements alone don't express directly.

### 4. Feature Selection

With 14 features available, we use `SelectKBest` with the chi-squared test to score each feature's statistical relationship with the target and keep the top 10. Chi-squared is appropriate here because all feature values are non-negative.

### 5. Model Training and Evaluation

Each model goes through two full rounds of training:

**Round 1  All 14 features:** 5-fold cross-validation followed by a fixed 80/20 train-test split (`random_state=44`).

**Round 2  Top 10 selected features:** Exact same process and same split to keep the comparison fair.

For each model in each round, the evaluation function produces:
- A colour-coded bar chart of R², F1, and Accuracy (green ≥ 0.7, orange borderline, red < 0.5)
- A confusion matrix heatmap
- A full per-class classification report

Results are saved to `Results/No Feature Selection/` and `Results/With Feature Selection/` respectively.

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Running the Notebook

```bash
jupyter notebook Iris_Flower_Classification.ipynb
```

Make sure the `Data Visualization/`, `Results/No Feature Selection/`, and `Results/With Feature Selection/` folders exist before running, as the notebook saves plots to those paths.

```bash
mkdir -p "Data Visualization"
mkdir -p "Results/No Feature Selection"
mkdir -p "Results/With Feature Selection"
```