import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#print("Path to dataset files:", path)

path = kagglehub.dataset_download("shambhurajejagadale/student-performance-prediction-dataset")

data = pd.read_csv(os.path.join(path, "student_dataset_10000_rows.csv"))
print(data)


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
    cost_history = []
    cost_history.append(calculate_cost(X,y,w,b))
    for _ in range(iterations):
        dj_dw, dj_db = calculate_slope(X,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost_history.append(calculate_cost(X,y,w,b))
    return w,b,cost_history

def zscore_normalisation(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = zscore_normalisation(X)
n = X.shape[1]
w_in = np.zeros(n)
w,b,cost_history = gradient_descent(X_norm, y, w_in, 0, 1, 5)
user_input = input("Enter hours studied, attendance, average hours of sleep, previous score (comma-separated): ")
x_in = np.array([float(val) for val in user_input.split(",")])
x_in_norm = (x_in-mu)/sigma
print(model(w, x_in_norm, b))
#print(cost_history)
#print(calculate_cost(X_norm,y,w,b))

plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(w,b)")
plt.title("Learning Curve")
plt.show()