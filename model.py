
# File: model
# Author: Taylor Kramer
# Date: 03/02/24

# class Model includes everything related to the creation and manipulation of desired ML model

#importing neccessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Model:
    
    def __init__(self, dataclass, modeltype):
        self.predictors = ["Open", "High", "Low", "Close", "Volume"]
        self.data = dataclass.data
        self.model = self.initialize_model(modeltype)
    
    #TODO: print and plot data functions
    
    def initialize_model(self, modeltype):
        
        "intializes model based on the user's preferences"
        
        '''Why random forests: resistant to overfitting, run relatively quickly, and can 
        pick up nonlinear tendencies in the data.
        This function initializes a Random Forest Classifier Model.'''
        if modeltype == "RFC":
            return RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        
        '''TODO: Why ANN: Why KNN: Why LSTM:'''
    
    def train_model(self, dataclass, trainx, trainy):
        if trainx.empty == True and trainy.empty == True:
            traindata, testdata = dataclass.get_train_test(0, dataclass.data.shape[0]-100, dataclass.data.shape[0]-100, dataclass.data.shape[0])
            print(traindata.info())
            print(traindata.info())
            self.model.fit(traindata[self.predictors], traindata["Target"])
        else:
            self.model.fit(trainx, trainy)