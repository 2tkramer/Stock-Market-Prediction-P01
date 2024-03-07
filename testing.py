
# File: testing
# Author: Taylor Kramer
# Date: 03/02/24

# Class testing includes everything related to testing desired model

#%%

#importing neccessary libraries
from data import Data
from model import Model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class Testing:
    def __init__(self, modelclass, dataclass):
        self.predictors = ["Open", "High", "Low", "Close", "Volume"]
        self.model = modelclass.model
        self.data = dataclass.data

    def model_prec_score(self, test_data):
        
        '''Use the saved_trained_model to predict the targets in the test set of the data. 
        Utilize sklearn's precision_score function to determine what percentage of the time 
        the model correctly predicts what direction the market will go.'''
        
        t_predictions = self.model.predict(test_data[self.predictors])
        #print(t_predictions)
        prec_score = precision_score(test_data["Target"], t_predictions)
        print(prec_score)
        return prec_score

    def predict(self, modelclass, dataclass, train, test):
        
        '''predict: trains the model, uses the model to make predictions (these
        predictions are placed into a pandas Series for compatability), ensuring 
        that the indexes are consistent. Then the model predictions are concatenated with 
        the actual outcomes in the data and then this may be used for teting and 
        visualization of model effectiveness.'''
        
        modelclass.train_model(dataclass, train[self.predictors], train["Target"])
        predictions = modelclass.model.predict(test[self.predictors])
        predictions_Series = pd.Series(predictions, index = test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions_Series],axis=1)
        return combined
    
    def backtest(self, modelclass, dataclass, start=2500, step=250):
        
        '''backtest: takes in desired data, the machine learning model, the predictors as
        well as the starting number of days worth of data to train the model on. Given there 
        are typically 250 trading days/year, the function will start by training the model on
        10 years of data (2500 days) and incrementally add 1 more year of training data after
        every iteration to be used for predicting the values in the following year.'''
        
        all_predictions = []
    
        for i in range(start, dataclass.data.shape[0], step):
            train, test = dataclass.get_train_test(0, i, i, i+step)
            predictions = self.predict(modelclass, dataclass, train, test)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)
    
    
'''Begin Backtesting'''
calldata = Data("^GSPC") #initiates instance of Data class to import raw data
calldata.clean_data() #wrangles data
mydata = calldata.get_data()

mymodel = Model(mydata, "RFC")
mymodel.train_model(calldata, pd.DataFrame(), pd.DataFrame())

backtesting = Testing(mymodel, mydata)

bt_predictions = backtesting.backtest(mymodel, calldata)
print(bt_predictions)
print(bt_predictions["Predictions"].value_counts())
bt_score = precision_score(bt_predictions["Target"], bt_predictions["Predictions"])
print(bt_score) 
print(bt_predictions["Target"].value_counts()/bt_predictions.shape[0])

horizons = [2,5,60,250,1000]

backtesting = backtesting.horizons(horizons)


#--------Visualizing Outcomes-----------------------------------------------------

'''Starts by placing predictions into a pandas Series for compatibility, ensuring 
that the indexes are consistent. Then the model predictions are concatenated with 
the actual outcomes as seen in the data and then this may be plotted to visualize 
the effectiveness of the model.'''


#combined.plot()

# %%
