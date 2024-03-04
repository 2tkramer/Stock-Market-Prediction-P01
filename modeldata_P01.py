
# File: modeldata_P01
# Author: Taylor Kramer
# Date: 03/02/24

# Prompt: Create a machine learning model to predict the direction of the stock market
# Answering the question: Will the S&P500 go up or down tomorrow?

#--------Importing Libraries and Dataset for this project-------------------------

'''Imports daily historical data from yahoo finance which already exists in a 
pandas dataframe. From here, the ticker function only grabs the S&P 500 
data (GSPC). Then the history function asks to import all historical data.'''

from modelClass import Model
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

sp500 = yf.Ticker("^GSPC") 
sp500 = sp500.history(period="max")
#sp500.plot.line(y = "Close", use_index=True)

#--------Wrangling Data-----------------------------------------------------------

'''Data wrangling begins with removing undesired columns including the dividends 
and stock splits which would not be as useful for predicting the directionality 
of the stockmarket. Also removes the data from years prior to 1990 with the 
assumption that fundamental shifts may have occured which could result in
a skewed model.'''

sp500 = sp500.drop(["Dividends", "Stock Splits"], axis=1)
sp500 = sp500.truncate(before="1990-01-01 00:00:00-05:00")

'''Setting target variable by adding a column that lists the next day's closing 
price for each day. Then uses function up_or_down to create a column of boolean 
values describing the target = will the price go up (1) or down (0) the 
following day? '''

sp500["Next Day"] = sp500["Close"].shift(-1)

def up_or_down(row):
    if row["Close"] < row["Next Day"]:
        val = 1
    else:
        val = 0
    return val

sp500["Target"] = sp500.apply(up_or_down, axis=1)

#--------Develop Random Forest Classifier model-----------------------------------

'''why random forests: resistant to overfitting, run relatively quickly, and can 
pick up nonlinear tendencies in the data.
Starts by initializing the model using RFCmodel function in Model class. 
Keeping in mind that this is time series data, only the last 100 rows of data 
will be used for the test set while all prior data is used in the training set
Next, the predictors are choosen out of the available features
function returns initialized model along with definitions for train & test sets
Lastly, the model is trained on the train dataset.'''

#example creating a starting_model using Model Class to initialize
starting = Model()
starting_model = starting.RFCmodel()
db_size = sp500.shape[0]
st_train, st_test = starting.Get_Train_Test(sp500, 0, db_size-100, db_size-100, db_size)
#print(st_train.info())
#print(st_test.info())

#training starting_model
starting_model.fit(st_train[starting.predictors], st_train["Target"])

#for accessing the model from other files
dump(starting_model, 'saved_model.joblib')

#--------starting_model is complete, refer to testing_model_P01 for model improvement

