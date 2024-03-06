
# File: data
# Author: Taylor Kramer
# Date: 03/02/24

# class Data includes everything related with importing and wrangling data

'''Imports daily historical data from yahoo finance which already exists in a 
pandas dataframe. From here, the ticker function only grabs the defined ticke
(ex. S&P 500 data = GSPC). Then the history function asks to import all historical 
data.'''

from model import Model
import yfinance as yf
class Data:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.import_data()
        
    def get_data(self):
        return self.data
        
    def import_data(self):
        self.data = yf.Ticker(self.ticker)
        self.data = self.data.history(period="max")
        return self.data
        
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
        



