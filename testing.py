
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
    
    
'''Begin Backtesting'''
mydata = Data("^GSPC") #initiates instance of Data class to import raw data
mydata.clean_data() #wrangles data
outputdata = mydata.get_data()

mymodel = Model("RFC")
mymodel.train_model(mydata, pd.DataFrame(), pd.DataFrame())

backtesting = Testing()

bt_predictions = backtesting.backtest(mymodel, mydata, 0.5)
bt_score = precision_score(bt_predictions["Target"], bt_predictions["Predictions"])
print("prec score original: ", bt_score) 
print(bt_predictions["Target"].value_counts()/bt_predictions.shape[0])

pastdays = [2,5,60,250,1000]
mydata.add_trends(mymodel, pastdays)
#print(mydata.data.info())
#improving model
new_model = mymodel.initialize_model("RFC", 200, 50, 1)
new_predictions = backtesting.backtest(mymodel, mydata, 0.6)
print(new_predictions["Predictions"].value_counts())
print("prec score improved: ", precision_score(new_predictions["Target"], new_predictions["Predictions"]))

#--------Visualizing Outcomes-----------------------------------------------------

'''Starts by placing predictions into a pandas Series for compatibility, ensuring 
that the indexes are consistent. Then the model predictions are concatenated with 
the actual outcomes as seen in the data and then this may be plotted to visualize 
the effectiveness of the model.'''


#combined.plot()

# %%
