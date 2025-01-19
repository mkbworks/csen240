import pandas as pd
import numpy as np
import math

def get_training_data(mode):
    if mode == 0:
        xi = np.array([1, 2], dtype='float')
        yi = np.array([300, 500], dtype='float')
        w = float(0)
        b = float(0)
        return xi, yi, w, b
    else:
        td = pd.read_csv("training_data.csv")
        tds = td.head(10)
        xi = tds['Square_Footage'].to_numpy(dtype='float')
        yi = tds['Price'].to_numpy(dtype='float')
        w = float(1500)
        b = float(350000)
        return xi, yi, w, b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = float(0)
    for i in range(m):
        y_hat = (w * x[i]) + b
        diff = (y_hat - y[i])
        cost += (diff * diff)
    cost = cost / (2 * m)
    return cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = dj_db = float(0)
    for i in range(m):
        y_hat = (w * x[i]) + b
        dj_dw_i = (y_hat - y[i]) * x[i]
        dj_db_i = (y_hat - y[i]) 
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

if __name__ == "__main__":
    xi, yi, w, b = get_training_data(0)
    
    print(f"xi = {xi}")
    print(f"yi = {yi}")
    print(f"m = {xi.shape[0]}")
    print(f"w = {w}")
    print(f"b = {b}")

    EPOCH = 10000
    alpha = 0.01
    for i in range(1, EPOCH + 1):
        dj_dw, dj_db = compute_gradient(xi, yi, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        cost = compute_cost(xi, yi, w, b)
        if i % (math.floor(EPOCH / 10)) == 0:
            print(f"Iteration {i} :: Cost = {cost} :: dj_dw = {dj_dw} :: dj_db = {dj_db} :: w = {float(w)} :: b = {float(b)}")