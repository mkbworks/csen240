import pandas as pd
import math

class DataProcessing:
    """
    Class to fetch data from .csv file, clean and transform them into the desired format.
    """
    def __init__(self, file):
        """
        Constructor to create a new instance of 'DataProcessing'.
        :param file: Complete file path of the .csv file containing the dataset.
        """
        self.file = str(file).strip()
        df = pd.read_csv(self.file, parse_dates=True)
        total_rows = df.shape[0]
        training_rows = math.ceil(0.8 * total_rows)
        self.training_data = df[:training_rows]
        self.validation_data = df[training_rows:]
    
    def clean_data(self):
        """
        Function to clean the training and validation data to remove unwanted columns and change values to a format suitable for training the machine learning model.
        :returns: no value(s).
        """
        self.training_data['Price'] = self.training_data['Price'] / 10000
        self.training_data['Square_Footage'] = self.training_data['Square_Footage'] / 100

        # Clean price and square footage values for validation data
        self.validation_data['Price'] = self.validation_data['Price'] / 10000
        self.validation_data['Square_Footage'] = self.validation_data['Square_Footage'] / 100
    
    def get_training_data(self):
        """
        Function that returns the feature and target values for training the machine learning model.
        :returns: tuple (ndarray object of feature values, ndarray object of target values)
        """
        self.clean_data()
        xi = self.training_data['Square_Footage'].to_numpy(dtype='float')
        yi = self.training_data['Price'].to_numpy(dtype='float')
        return xi, yi