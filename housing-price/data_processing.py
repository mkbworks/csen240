import pandas as pd
import numpy as np
import math

class DataProcessing:
    """
    Class to fetch data from .csv file, clean and transform them into the desired format.
    """
    def __init__(self, op_col, *ip_cols):
        """
        Fetches data from the CSV file, extracts the training data (80% of total data available), and parses the input features and output values as an instance of DataProcessing class.
        :param op_col: a string value containing the name of the output column in the CSV file.
        :param *ip_cols: a list of strings representing all the input features in the CSV file.
        """
        df = pd.read_csv("housing.csv", parse_dates=True)
        total_rows = df.shape[0]
        self.train_rowc = math.ceil(0.8 * total_rows)
        trd = df[:self.train_rowc]
        self.yi = trd[op_col].to_numpy(dtype='float')
        if len(ip_cols) == 1:
            self.xi = trd[ip_cols[0]].to_numpy(dtype='float')
        elif len(ip_cols) > 1:
            cols = []
            for i in range(len(ip_cols)):
                cols.append(trd[ip_cols[i]].to_numpy(dtype='float'))
            self.xi = np.column_stack(tuple(cols))
        
    def normalize_features(self):
        """
            Normalizes the input features using z-score normalization.
            :returns: a ndarray containing the normalized feature values.
        """
        if self.xi.ndim == 1:
            xi_mean = np.mean(self.xi)
            xi_std = np.std(self.xi)
            xi_nm = (self.xi - xi_mean) / xi_std
            return xi_nm
        else:
            mean_values = np.mean(self.xi, axis=0)
            std_values = np.std(self.xi, axis=0)
            xi_nm = (self.xi - mean_values) / std_values
            return xi_nm
    
    def normalize_targets(self):
        """
            Normalizes the output target values using z-score normalization.
            :returns:  a ndarray containing the normalized target values.
        """
        yi_mean = np.mean(self.yi)
        yi_std = np.std(self.yi)
        yi_nm = (self.yi - yi_mean) / yi_std
        return yi_nm