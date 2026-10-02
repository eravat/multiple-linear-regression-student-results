# Student Performance Prediction — Multiple Linear Regression from Scratch

A multiple linear regression model, implemented from scratch using vectorized gradient descent (no scikit-learn), predicting student exam scores from several input features.

## Overview

This project extends my earlier univariate gradient descent implementation to multiple features. The goal was to implement vectorized multiple linear regression by hand — the cost function, the gradient computation, and the update rule — using NumPy matrix operations, rather than relying on a library's built-in `.fit()` method.

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

### Feature scaling (z-score normalization)

Before training, each feature is normalized using z-score normalization:

```
x_norm = (x - mean) / std
```

This rescales every feature to have a mean of 0 and a standard deviation of 1, so features on very different raw scales (e.g. hours studied vs. previous exam score out of 100) contribute to the gradient fairly, rather than one feature's larger raw values dominating. It also allows a much larger learning rate to be used, which sped up convergence dramatically compared to the unscaled univariate project.

The mean and standard deviation used for scaling are computed from the training data only, and the same values are reused to scale any new input before making a prediction — the model was trained on scaled features, so new inputs must be transformed the same way to produce a meaningful prediction.

### Learning curve

Cost is recorded at every iteration during training and plotted against iteration number afterward, to visually confirm gradient descent is converging (cost decreasing and flattening out) rather than diverging or oscillating. See the plot in the Results section below.

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

- I learned how effective vectorization is in speeding up gradient descent compared to my last univariate linear regression, which took longer to converge even though the dataset was smaller and I used fewer features.
- I learned how much feature scaling (z-score normalization) matters for gradient descent — before scaling, the learning rate had to be extremely small (`1e-7`) to avoid diverging; after scaling, a much larger learning rate converged in only a handful of iterations.

## Limitations

- The model can predict exam scores above 100% (or below 0%) for inputs far from the training data's range. Linear regression has no built-in output bounds — it fits a straight line and extrapolates it indefinitely, with no concept that scores are naturally capped between 0 and 100. This is most likely to happen for feature combinations outside the range the model was trained on. A production version of this model would need to clip predictions to a valid range, or use a model better suited to bounded outputs.
