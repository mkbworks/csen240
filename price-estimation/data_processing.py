import pandas as pd
import math

class DataProcessing:
    def __init__(self, file):
        self.file = str(file).strip()
        df = pd.read_csv(self.file, parse_dates=True)
        total_rows = df.shape[0]
        training_rows = math.ceil(0.8 * total_rows)
        self.training_data = df[:training_rows]
        self.validation_data = df[training_rows:]
    
    def clean_data(self):
        # Clean price and square footage values for training data
        self.training_data['Price'] = self.training_data['Price'] / 10000
        self.training_data['Square_Footage'] = self.training_data['Square_Footage'] / 100

        # Clean price and square footage values for validation data
        self.validation_data['Price'] = self.validation_data['Price'] / 10000
        self.validation_data['Square_Footage'] = self.validation_data['Square_Footage'] / 100
    
    def get_training_data(self):
        self.clean_data()
        xi = self.training_data['Square_Footage'].to_numpy(dtype='float')
        yi = self.training_data['Price'].to_numpy(dtype='float')
        return xi, yi