
# File: data
# Author: Taylor Kramer
# Date: 03/02/24

# class Data includes everything related to importing, wrangling, and manipulation of data

'''Imports daily historical data from yahoo finance which already exists in a 
pandas dataframe. From here, the ticker function only grabs the defined ticke
(ex. S&P 500 data = GSPC). Then the history function asks to import all historical 
data.'''

import yfinance as yf
class Data:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.import_data()
        
    def import_data(self):
        self.data = yf.Ticker(self.ticker)
        self.data = self.data.history(period="max")
        return self.data
    
    def get_data(self):
        return self.data
    
    def get_train_test(self, a, b, c, d):
        
        ''' Creates training and testing datasets from self.data using the inputs which
        define the starting and ending row for the train set and the starting and ending 
        row for the test set.'''
        
        train = self.data.iloc[a:b]
        test = self.data.iloc[c:d]
        return train, test
        
    def up_or_down(row):
        if row["Close"] < row["Next Day"]:
            val = 1
        else:
            val = 0
        return val
        
    def clean_data(self):

        '''Data wrangling begins with removing undesired columns including the dividends 
        and stock splits which would not be as useful for predicting the directionality 
        of the stockmarket. Also removes the data from years prior to 1990 with the 
        assumption that fundamental shifts may have occured which could result in
        a skewed model.'''

        self.data = self.data.drop(["Dividends", "Stock Splits"], axis=1)
        self.data = self.data.truncate(before="1990-01-01 00:00:00-05:00")

        '''Setting target variable by adding a column that lists the next day's closing 
        price for each day. Then uses function up_or_down to create a column of boolean 
        values describing the target = will the price go up (1) or down (0) the 
        following day? '''

        self.data["Next Day"] = self.data["Close"].shift(-1)
        
        def up_or_down(row):
            if row["Close"] < row["Next Day"]:
                val = 1
            else:
                val = 0
            return val
        
        self.data["Target"] = self.data.apply(up_or_down, axis=1)
        self.data = self.data.dropna()
        
    def add_trends(self, modelclass, pastdays):
    
        '''horizons: adding more columns/predictors based on trends for the model to make 
        better predictions. The for loop iterates through horizons, an array which holds 
        numbers that represent number of past days (num). For each iteration, the average 
        closing price for the last "num" days is calculatede. Then a new column of data is 
        created "Close_Ratio_num" which creates a relationship between the closing price 
        of the present day and the avg price of the past "num" days. Each iteration also 
        creates a "Trend_num" column which contains the sum of days in the last "num" of 
        days where the stock market went up, helping detect a trend in the market. This 
        finally adds the two new columns to the predictors array through each iteration.'''
    
        for num in pastdays:
            rolling_averages = self.data.rolling(num).mean()
            ratio_column = f"Clos_Ratio_{num}"
            self.data[ratio_column] = self.data["Close"] / rolling_averages["Close"] 
            trend_column = f"Trend_{num}" 
            self.data[trend_column] = self.data.shift(1).rolling(num).sum()["Target"]
            modelclass.predictors += [ratio_column, trend_column]
            self.data = self.data.dropna()
        return
    



