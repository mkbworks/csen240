# Machine Learning

To run the python scripts present in this repository, the following must be installed in your system.

- Python (>= 3.12) or higher
- `pip` (package manager for python)

## Virtual Environment

To run the project locally, set up a virtual environment on the home directory of the project. To create a `venv` in python, run the following command.

```bash
chmod +x venv.sh

# This shell script creates and activates a virtual environment in the current directory.
# It also activates the venv and installs all the requirements for the project.
./venv.sh
```

To deactivate the venv, run the below command on the terminal.

```bash
deactivate
```

## Jupyter Notebooks

### housing_price_univariate.ipynb

This notebook file estimates the price of a house based on the area of the house (measured in sq ft). It employs univariate linear regression model to train and predict the target values.

- The input values are regularized using L2 (Lasso) regularization to avoid overfitting.
- It uses `batch gradient descent` algorithm to minimise the model's empirical loss and find the corresponding model parameters.

### housing_price_multivariate.ipynb

This notebook file predicts the price of a house based on multiple factors namely

- size of the house
- Number of bathrooms
- Number of bedrooms
- Number of stories
- Number of parking lots
- If house is on a mainroad
- If it has a guestroom, air conditioning, basement, hot water heating
- Furnishing status

It employs multivariate linear regression model to make predictions with lasso regularized inputs.