import numpy as np

class Normalization:
    @staticmethod
    def min_max(x):
        xmin = np.min(x, axis=0)
        xmax = np.max(x, axis=0)
        return (x - xmin) / (xmax - xmin)

    @staticmethod
    def zscore(x):
        xmean = np.mean(x, axis=0)
        xstd = np.std(x, axis=0)
        return (x - xmean) / xstd