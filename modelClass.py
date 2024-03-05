from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Model:
    
    def __init__(self, data):
        self.predictors = ["Open", "High", "Low", "Close", "Volume"]
        self.data = data
    
    def Get_Train_Test(self, a, b, c, d):
        
        ''' Creates training and testing datasets from self.data using the inputs which
        define the starting and ending row for the train set and the starting and ending 
        row for the test set.'''
        
        train = self.data.iloc[a:b]
        test = self.data.iloc[c:d]
        return train, test
        
    def RFCmodel(self):
        
        '''Why random forests: resistant to overfitting, run relatively quickly, and can 
        pick up nonlinear tendencies in the data.
        This function initializes a Random Forest Classifier Model.'''
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        return model

    def predict(self, train, test, model):
        
        '''predict: trains the model, uses the model to make predictions (these
        predictions are placed into a pandas Series for compatability), ensuring 
        that the indexes are consistent. Then the model predictions are concatenated with 
        the actual outcomes in the data and then this may be used for teting and 
        visualization of model effectiveness.'''
        
        model.fit(train[self.predictors], train["Target"])
        predictions = model.predict(test[self.predictors])
        predictions_Series = pd.Series(predictions, index = test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions_Series],axis=1)
        return combined
    
    def backtest(self, model, start=2500, step=250):
        
        '''backtest: takes in desired data, the machine learning model, the predictors as
        well as the starting number of days worth of data to train the model on. Given there 
        are typically 250 trading days/year, the function will start by training the model on
        10 years of data (2500 days) and incrementally add 1 more year of training data after
        every iteration to be used for predicting the values in the following year.'''
        
        all_predictions = []
    
        for i in range(start, self.data.shape[0], step):
            train, test = self.Get_Train_Test(0, i, i, i+step)
            predictions = self.predict(train, test, model)
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