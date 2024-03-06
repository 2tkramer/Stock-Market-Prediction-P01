
# File: testing
# Author: Taylor Kramer
# Date: 03/02/24

# everything involving the testing of desired model

#%%

#importing neccessary libraries
from data import Data
from model import Model
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#importing trained model from file

saved_trained_model = load('saved_model.joblib')

#--------Testing Model------------------------------------------------------------

'''Use the saved_trained_model to predict the targets in the test set of the data. 
Utilize sklearn's precision_score function to determine what percentage of the time 
the model correctly predicts what direction the market will go.'''

#testing_model = Model(sp500)
#t_predictions = saved_trained_model.predict(st_test[testing_model.predictors])
#print(t_predictions)
#test_score = precision_score(st_test["Target"], t_predictions)
#print(test_score)

'''Begin Backtesting'''
calldata = Data("^GSPC") #initiates instance of Data class to import raw data
calldata.clean_data() #wrangles data
sp500 = calldata.get_data()

backtesting = Model(sp500, "RFC")
bt_predictions = backtesting.backtest()
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
