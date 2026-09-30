# Student Performance Prediction — Multiple Linear Regression from Scratch

A multiple linear regression model, implemented from scratch using vectorized gradient descent (no scikit-learn), predicting student exam scores from several input features.

## Overview

This project extends an earlier univariate gradient descent implementation ([link to that repo]) to multiple features. The goal was to implement vectorized multiple linear regression by hand — the cost function, the gradient computation, and the update rule — using NumPy matrix operations, rather than relying on a library's built-in `.fit()` method.

## Dataset

- **Source:** [Student Performance Prediction Dataset](https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset) (Kaggle), downloaded automatically via `kagglehub`.
- **Features used:** [list your actual feature columns here, e.g. Hours_Studied, Attendance, Previous_Scores]
- **Target:** [your target column, e.g. Exam_Score]

## Method

The model fits:

```
f(x) = w·x + b
```

where `w` is a vector of weights (one per feature) and `x` is a vector of feature values for a single student. `w` and `b` are learned via **vectorized batch gradient descent**:

1. Compute all predictions at once: `X @ w + b`
2. Compute the cost function `J(w, b)` — mean squared error across all training examples
3. Compute the gradient vector `∂J/∂w` (one partial derivative per feature) and `∂J/∂b`, using `X.T @ error`
4. Simultaneously update all weights and `b` using the learning rate `α`
5. Repeat until convergence

No explicit loops over training examples or features — every step uses NumPy matrix/vector operations.

## Files

- `main.py` — loads data, runs gradient descent, prints/plots results
- `requirements.txt` — Python dependencies

## How to run

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python main.py
```

**Note:** the dataset is downloaded automatically via `kagglehub` the first time the script runs (no manual download needed). This requires a Kaggle account with API credentials configured — see [kagglehub's setup instructions](https://github.com/Kaggle/kagglehub) if you haven't used it before. The dataset is cached locally after the first run.


## What I learned

- I learned how effective vectorization is in speeding up gradient descent compared to my last univariate linear regression which took longer to converge even though the dataset was smaller and I used less features.

## Next steps

- Try polynomial regression on one or more features to check for non-linear relationships
- Compare results against scikit-learn's `LinearRegression()` to validate the from-scratch implementation