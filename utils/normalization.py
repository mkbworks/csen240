import numpy as np

class Normalization:
    """
    Class that serves as a container for functions that normalize or standardize inputs for machine learning.
    """
    @staticmethod
    def min_max(x):
        """
        Performs min-max normalization to the given inputs and returns the normalized values.
        x - (m x n) matrix holding all the input values.
        returns: (m x n) matrix containing the normalized values.
        """
        xmin = np.min(x, axis=0)
        xmax = np.max(x, axis=0)
        return (x - xmin) / (xmax - xmin)

    @staticmethod
    def zscore(x):
        """
        Performs z-score standardization to the given inputs and returns the normalized values.
        x - (m x n) matrix holding all the input values.
        returns: (m x n) matrix containing the normalized values.
        """
        xmean = np.mean(x, axis=0)
        xstd = np.std(x, axis=0)
        return (x - xmean) / xstd