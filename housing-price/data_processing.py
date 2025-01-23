import pandas as pd
import math

class DataProcessing:
    """
    Class to fetch data from .csv file, clean and transform them into the desired format.
    """
    def __init__(self):
        """
        Constructor to create a new instance of 'DataProcessing'. The first 80% of rows are allotted as training data and the rest 20% are allotted as validation data.
        """
        df = pd.read_csv("housing.csv", parse_dates=True)
        total_rows = df.shape[0]
        training_rows = math.ceil(0.8 * total_rows)
        self.training_data = df[:training_rows]
        self.validation_data = df[training_rows:]
    
    def clean_data(self):
        """
        Function to clean the training and validation data to remove unwanted columns and change values to a format suitable for training the machine learning model.
        :returns: no value(s).
        """
        self.training_data['area'] = self.training_data['area'] / 1000
        self.training_data['price'] = self.training_data['price'] / 1000000

        # Clean price and square footage values for validation data
        self.validation_data['area'] = self.validation_data['area'] / 1000
        self.validation_data['price'] = self.validation_data['price'] / 1000000
    
    def get_training_data(self):
        """
        Function that returns the feature and target values for training the machine learning model.
        :returns: tuple (ndarray object of feature values, ndarray object of target values)
        """
        self.clean_data()
        xi = self.training_data['area'].to_numpy(dtype='float')
        yi = self.training_data['price'].to_numpy(dtype='float')
        return xi, yi