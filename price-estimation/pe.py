# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = float(0)
    for i in range(m):
        y_hat = (w * x[i]) + b
        cost += (y_hat - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

# %%
def compute_gradient(x, y, w, b):
    breakpoint()
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
    print(f"m = {m}, dj_dw = {dj_dw}")
    print(f"dj_db = {dj_db}")
    return dj_dw, dj_db

# %%
td = pd.read_csv("training_data.csv")
tds = td.head(10)
xi = tds['Square_Footage'].to_numpy(dtype='float')
yi = tds['Price'].to_numpy(dtype='float')
print(f"xi = {xi}")
print(f"yi = {yi}")
print(f"m = {xi.shape[0]}")

# %%
plt.plot(xi, yi, label='Price vs Square Footage')
plt.title("Price Estimation - Training data")
plt.xlabel("Square Footage")
plt.ylabel("Price (in $)")
plt.grid()
plt.show()

# %%
w = b = float(0)
EPOCH = 100
alpha = 0.01
wbs = list()
jwbs = list()
for i in range(1, EPOCH + 1):
    dj_dw, dj_db = compute_gradient(xi, yi, w, b)
    w = w - (alpha * dj_dw)
    b = b - (alpha * dj_db)
    cost = compute_cost(xi, yi, w, b)
    wbs.append((w, b))
    jwbs.append(cost)
    if i % 1000 == 0:
        print(f"At end of iteration {i} :: Cost is {cost} :: Model parameters are {(w, b)}")

# %%
plt.plot(wbs, jwbs, label="Cost vs model parameters")
plt.xlabel("Model parameters - (w, b)")
plt.ylabel("Cost function for linear regression")
plt.grid()
plt.show()


