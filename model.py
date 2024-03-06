
# File: model
# Author: Taylor Kramer
# Date: 03/02/24

# everything involving the creation and manipulation of desired ML model

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Model:
    
    def __init__(self, data, modeltype):
        self.predictors = ["Open", "High", "Low", "Close", "Volume"]
        self.data = data
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
    
    #TODO: fix function
    def train_model(self, trainx, trainy):
        
        if trainx == None and trainy == None:
            trainx, trainy = self.get_train_test(0, self.data.shape[0]-100, self.data.shape[0]-100, self.data.shape[0])
            print(trainx.info())
            print(trainy.info())
        
        self.model.fit(trainx[self.predictors], trainy["Target"])
    
    def get_train_test(self, a, b, c, d):
        
        ''' Creates training and testing datasets from self.data using the inputs which
        define the starting and ending row for the train set and the starting and ending 
        row for the test set.'''
        
        train = self.data.iloc[a:b]
        test = self.data.iloc[c:d]
        return train, test

    def predict(self, train, test):
        
        '''predict: trains the model, uses the model to make predictions (these
        predictions are placed into a pandas Series for compatability), ensuring 
        that the indexes are consistent. Then the model predictions are concatenated with 
        the actual outcomes in the data and then this may be used for teting and 
        visualization of model effectiveness.'''
        self.train_model(train[self.predictors], train["Target"])
        predictions = self.model.predict(test[self.predictors])
        predictions_Series = pd.Series(predictions, index = test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions_Series],axis=1)
        return combined
    
    def backtest(self, start=2500, step=250):
        
        '''backtest: takes in desired data, the machine learning model, the predictors as
        well as the starting number of days worth of data to train the model on. Given there 
        are typically 250 trading days/year, the function will start by training the model on
        10 years of data (2500 days) and incrementally add 1 more year of training data after
        every iteration to be used for predicting the values in the following year.'''
        
        all_predictions = []
    
        for i in range(start, self.data.shape[0], step):
            train, test = self.get_train_test(0, i, i, i+step)
            predictions = self.predict(train, test)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)
    
    def horizons(self, horizons):
        
        '''adding more columns/predictors based on trends for the model to make better predictions.
        The for loop iterates through horizons, an array which holds numbers that represent number 
        of past days (num). For each iteration, the average closing price for the last "num" days is 
        calculated. Then a new column of data is created "Close_Ratio_num" which creates a relationship 
        between the closing price of the present day and the avg price of the past "num" days. Each 
        iteration also creates a "Trend_num" column which contains the sum of days in the last "num" 
        of days where the stock market went up, helping detect a trend in the market. This finally adds 
        the two new columns to the predictors array through each iteration.'''
       
        for num in horizons:
            rolling_averages = self.data.rolling(num).mean()
            ratio_column = f"Close_Ratio_{num}"
            self.data[ratio_column] = self.data["Close"] / rolling_averages["Close"] 
            trend_column = f"Trend_{num}" 
            self.data[trend_column] = self.data.shift(1).rolling(num).sum()["Target"]
            self.predictors += [ratio_column, trend_column]
            self.data = self.data.dropna()
        return