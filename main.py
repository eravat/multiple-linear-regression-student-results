import kagglehub
import os
import pandas as pd
import numpy as np

#print("Path to dataset files:", path)

path = kagglehub.dataset_download("shambhurajejagadale/student-performance-prediction-dataset")

data = pd.read_csv(os.path.join(path, "student_dataset_10000_rows.csv"))


'''columns: 'study_hours', 'attendance', 'sleep_hours', 'internet_usage',
       'assignments_completed', 'previous_score', 'exam_score',
       'placement_status' '''

X = data[["study_hours", "attendance", "sleep_hours", "previous_score"]].values
y = data["exam_score"].values
#print(x.shape, y.shape)    printed (10000,4), (10000,)

#functions: model, calulate_slope, calculate_cost, gradient_descent

def model(w_in, x_in, b_in):
    predicted_score = np.dot(x_in, w_in) + b_in
    return predicted_score
    
def calculate_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b     #used vectorisation to compute dot product of all rows of x with vector w at once
    errors = predictions - y
    squared_errors = errors**2
    cost = (np.sum(squared_errors))/(2*m)
    return cost

def calculate_slope(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    dj_dw = (X.T @ errors)/m    #transposes x so that each single feature accross all 10k training exapmples is displayed as a row, then can compute dot product of each feature with all error values giving array of differentiated cost function wrt each weighting
    dj_db = (np.sum(errors))/m
    return dj_dw, dj_db         #dj_dw[i] is differential of cost wrt. w[i]

def gradient_descent(X, y, w_init, b_init, alpha, iterations):
    w = w_init
    b = b_init
    for _ in range(iterations):
        dj_dw, dj_db = calculate_slope(X,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
    return w,b

def zscore_normalisation(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = zscore_normalisation(X)
n = X.shape[1]
w_in = np.zeros(n)
w,b = gradient_descent(X_norm, y, w_in, 0, 1, 5)
x_in = np.array([7,56,8,70])
x_in_norm = (x_in-mu)/sigma
print(model(w, x_in_norm, b))
#print(calculate_cost(X_norm,y,w,b))