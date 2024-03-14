
# File: testing
# Author: Taylor Kramer
# Date: 03/02/24

# Class testing includes everything related to testing desired model

#importing neccessary libraries
from data import Data
from model import Model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class Testing:
    
    def __init__(self):
        # here if needed
        return

    def model_prec_score(self, dataclass, modelclass):
        
        '''Use the saved_trained_model to predict the targets in the test set of the data. 
        Utilize sklearn's precision_score function to determine what percentage of the time 
        the model correctly predicts what direction the market will go.'''
        
        traindata, testdata = dataclass.get_train_test(0, dataclass.data.shape[0]-100, dataclass.data.shape[0]-100, dataclass.data.shape[0])
        t_predictions = modelclass.model.predict(testdata[modelclass.predictors])
        prec_score = precision_score(testdata["Target"], t_predictions)
        return prec_score

    def predict(self, modelclass, dataclass, train, test, confidence):
        
        '''predict: trains the model, uses the model to make predictions (these
        predictions are placed into a pandas Series for compatability), ensuring 
        that the indexes are consistent. Then the model predictions are concatenated with 
        the actual outcomes in the data and then this may be used for teting and 
        visualization of model effectiveness. TODO: explain confidence & predict_proba'''

        if modelclass.new_predictors == []:
            predictors = modelclass.predictors
        else:
            predictors = modelclass.new_predictors
        
        modelclass.train_model(dataclass, train[predictors], train["Target"])
        preds = modelclass.model.predict_proba(test[predictors])[:,1]
        for i in range(len(preds)):
            if preds[i] >= confidence:
                preds[i] = 1
            else:
                preds[i] = 0
        predictions_Series = pd.Series(preds, index = test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions_Series],axis=1)
        return combined
    
    def backtest(self, modelclass, dataclass, confidence, start=2500, step=250):
        
        '''backtest: takes in desired data, the machine learning model, the predictors as
        well as the starting number of days worth of data to train the model on. Given there 
        are typically 250 trading days/year, the function will start by training the model on
        10 years of data (2500 days) and incrementally add 1 more year of training data after
        every iteration to be used for predicting the values in the following year.'''
        
        all_preds = []
        for i in range(start, dataclass.data.shape[0], step):
            train, test = dataclass.get_train_test(0, i, i, i+step)
            preds = self.predict(modelclass, dataclass, train, test, confidence)
            all_preds.append(preds)
        return pd.concat(all_preds)