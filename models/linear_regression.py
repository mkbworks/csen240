import copy
import numpy as np

"""
Class containing methods to create and train a linear regression model with 'n' features and 'm' training samples.
"""
class LinearRegression:
    def __init__(self, alpha, epochs, lambda_r):
        """
        Instantiate a new instance of linear regression model by providing the following parameters:
        alpha - the learning rate of the model.
        epochs - the number of training iterations to be used for iterative algorithms like gradient descent.
        lambda_r - hyperparameter value for L2 regularization.
        """
        self.alpha = alpha
        self.epochs = epochs
        self.w_final = None
        self.cost_final = float('inf')
        self.lambda_r = lambda_r

    def compute_cost(self, x, y, w):
        """
        Computes the average loss (aka empirical risk) for the linear regression model.
        x - (m x n) matrix containing all the training samples. Each row represents a training sample.
        y - (m x 1) matrix containg the target values for the training samples.
        w - (n x 1) matrix containing the weights for the linear regression model.
        returns: scalar value representing the average loss for the given training samples and model parameters.
        """
        y_hat = np.dot(x, w)
        cost = np.mean(np.square(y_hat - y))
        cost = cost / 2
        cost = cost + ((self.lambda_r / 2) * np.mean(np.square(w)))
        return cost

    def compute_gradient(self, x, y, w):
        """
        Computes the factor by which the model parameters have to be decreased to minimise the empirical risk.
        x - (m x n) matrix containing all the training samples. Each row represents a training sample.
        y - (m x 1) matrix containg the target values for the training samples.
        w - (n x 1) matrix containing the weights for the linear regression model.
        returns: a tuple with 2 values
            - first value is a (m x 1) matrix denoting the delta by which the model parameters have to be decreased.
            - second value is a scalar denoting the delta by which the model bias has to be decreased.
        """
        m = x.shape[0]
        y_hat = np.dot(x, w)
        diff = y_hat - y
        dj_dw = ((1 / m) * np.dot(x.T, diff)) + ((self.lambda_r / m) * w)
        return dj_dw

    def train_gd(self, x, y):
        """
        Trains the linear regression model with the given training feature and target values using gradient descent.
        x - (m x n) matrix containing all the training samples. Each row represents a training sample.
        y - (m x 1) matrix containg the target values for the training samples.
        returns: a list of all the cost values computed for every epoch.
        """
        n = x.shape[1]
        np.random.seed(20)
        w = np.random.randn(n, 1)
        cost_values = []
        for index in range(1, self.epochs + 1):
            dj_dw = self.compute_gradient(x, y, w)
            w = w - self.alpha * dj_dw
            cst = self.compute_cost(x, y, w)
            cost_values.append(cst)
            if cst < self.cost_final:
                self.cost_final = cst
                self.w_final = np.copy(w)
        return cost_values

    def train_ols(self, x, y):
        """
        Trains the linear regression model using the closed form solutions and determines the best value of the model parameters.
        x - (m x n) matrix containing all the training samples. Each row represents a training sample.
        y - (m x 1) matrix containg the target values for the training samples.
        returns: None
        """
        xtx = np.dot(x.T, x)
        xtxinv = np.linalg.inv(xtx)
        self.w_final = np.linalg.multi_dot([xtxinv, x.T, y])

    def validate(self, x, y):
        """
        Validates the model using the given validation feature, target values and the finalized model parameters.
        x - (m x n) matrix representing the validation samples.
        y - (m x 1) matrix containg the target values for the validation samples.
        returns: a scalar value denoting the average loss for the validation dataset.
        """
        w = np.copy(self.w_final)
        cst = self.compute_cost(x, y, w)
        return cst

    def predict(self, x):
        """
        Computes the predicted target values for the given samples with the minmized model parameters.
        x - (m x n) matrix representing the sample values.
        returns: (m x 1) matrix containing the predicted target values.
        """
        return np.dot(x, self.w_final)

    def r2_score(self, x, y):
        """
        Computes the adjusted R-2 score for the given samples with the minmized model parameters.
        x - (m x n) matrix representing the sample values.
        y - (m x 1) matrix containg the target values for the samples.
        returns: scalar value between 0 and 1, indicating how well the model predicts the target values. Higher the value, the better.
        """
        m, n = x.shape
        y_hat = self.predict(x)
        y_mean = np.mean(y)
        ss_res = np.sum(np.square(y - y_hat))
        ss_tot = np.sum(np.square(y - y_mean))
        r2_score = 1 - (ss_res / ss_tot)
        r2_adj = 1 - (((1 - (r2_score ** 2)) * (m - 1))/ (m - n - 1))
        return r2_adj