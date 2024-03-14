# File: data
# Author: Taylor Kramer
# Date: 03/12/24

# class Interact includes everything related to the interactions in main.py

'''Prompt: Creating and testing machine learning models to predict the direction of stocks.
   Answering the question: Will a certain stock go up or down tomorrow? How well can 
   various machine learning models make these predictions.'''

#importing neccessary libraries
from data import Data
from model import Model
from testing import Testing
import pandas as pd
import csv
import matplotlib.pyplot as plt
from joblib import dump # for pickling model
from sklearn.metrics import precision_score

class Interact:
    
    def __init__(self):
        self.repeat = 'yes'
        self.reprompt_ticker = 'yes'
        self.all_tickers = self.get_ticker_list()
        self.count = 0

    def get_ticker_list(self):
        
        '''places all yahoo finance tickers into a flattened list to be analyzed later in the program.'''
        
        with open(r"C:\Users\Taylor\Documents\GitRepos\Stock-Market-Prediction-P01\YahooTickerSymbols.csv") as file:
            reader = csv.reader(file)
            ticker_list = list(reader)
        flattened_list = []
        for row in ticker_list:
            flattened_list.extend(row)
            
        return flattened_list
    
    def user_interface(self):
        
        '''this function provides everthing that the user interacts with throughout the program and 
        also responds to all the inputs.'''
        
        while self.repeat == 'yes':
   
            #--------Data-----------------------------------------------------------------------------------
            
            print("\nHowdy! :) What yahoo finance data are you looking to analyze today?")
            print("Here are some examples: ^GSPC (S&P 500), ^DJI (Dow Jones), AAPL (Apple), TSLA (Tesla)\n")
            while self.reprompt_ticker == 'yes':
                self.count += 1
                ticker = input("Your ticker choice: ")
                self.study_tickers(ticker)
            mydata = Data(ticker) #initiates instance of Data class -> imports raw data using choosen ticker
            mydata.clean_data() #wrangles data
            outputdata = mydata.get_data() #places cleaned data into variable: mydata
            outputdata.plot.line(y="Close", use_index=True) #visualize market data
            plt.title('Quick Look at Data')
            plt.xlabel('Year')
            plt.ylabel('Closing Price')
            plt.show()
    
            #--------Model----------------------------------------------------------------------------------
            
            #TODO: start adding different types of machine learning models
            print("\nGreat choice!\nNow, what type of ML model are you looking to use?")
            print("Here are the options we have available as of right now: RFC (Randon Forest Classifier)\n")
            modeltype = input("Your choice: ")
            #TODO: reprompt if choice doesn't exist, have the reprompt give tips
            print("\nAwesome! I am initializing and training your model on your chosen dataset now...\n")
            mymodel = Model(modeltype) #initiates instance of Model class -> initializes model w/ basic parameters TODO: ask about updating parameters later
            mymodel.train_model(mydata, pd.DataFrame(), pd.DataFrame()) #trains base model, 
            #TODO: create visualizations for the initial trained model
            #precision result
            print("All Done! Now time to do a little bit of testing.")
            
            #--------Testing & Improvement------------------------------------------------------------------

            print("\nLet's do some initial backtesting... starting now...")
            backtesting = Testing()
            predictions_a = backtesting.backtest(mymodel, mydata, 0.5)
            bt_score = precision_score(predictions_a["Target"], predictions_a["Predictions"])
            print("\nHere is the precision score for the raw untouched model: ", bt_score,"\n") 
            #TODO: have the previous line print out the precision score in a percentage with 1 decimal point
            #print(predictions_a["Target"].value_counts()/predictions_a.shape[0])
            
            # improving model
            
            print("Now, let's do some things to try and improve our model!")
            print("We will start by creating new predictors that relate each day's closing price to the average closing price of a previous number of days.")
            pastdays = [2,5,60,250,1000]
            mydata.add_trends(mymodel, pastdays)
            predictions_b = backtesting.backtest(mymodel, mydata, 0.5)
            print("\nPrecision score with updated predictors: ", precision_score(predictions_b["Target"], predictions_b["Predictions"]))
            #TODO: allow user to adjust this to their liking and show updated precision scores 
            
            print("\nNext, let's try setting different parameters when initializing the model")
            mymodel.model = mymodel.initialize_model("RFC", 200, 50, 1)
            predictions_c = backtesting.backtest(mymodel, mydata, 0.5)
            #print(predictions_c["Predictions"].value_counts())
            print("\nPrecision score with updated parameters: ", precision_score(predictions_c["Target"], predictions_c["Predictions"]))
            #TODO: allow user to adjust this as well
            
            print("\nFinally, let's play around with setting different confidence levels") #TODO: exlpain confidence levels
            predictions_d = backtesting.backtest(mymodel, mydata, 0.6) #ex: 60%
            print("\nPrecision score with updated confidence level: ", precision_score(predictions_d["Target"], predictions_d["Predictions"]))
            #TODO: allow user to adjust this as well
            
            
            '''Things to help improve model: correlating indices that are open over night before the
            US markets are open, correlate the news(NLP?), increase resolution: hourly or minute data'''

            #--------Finish---------------------------------------------------------------------------------
            print("\nWe are all done!\n")
            print("Would you like to save this model?\n")
            save_model = input("'yes' or 'no': ")
            if save_model == "yes":
                dump(mymodel, 'saved_model.joblib') 
                #TODO: have new file name depend on data and type of model # pickles finalized model for use in the future
                print("\nYay! Your model has been pickled and saved into a file called: _____\n")
                print("HINT: to use the pickled model elsewhere: from joblib import load and then savedmodel = load('mymodel.joblib')\n")
            else:
                print("Okay! ")
            print("Would you like to try anything else today?\n")
            self.repeat = input("'yes' or 'no': ")
        
        return
    
    def study_tickers(self, str):
        
        '''determines if the yahoo finance ticker that the user provided exists and if not provides hints 
        and examples of tickers based on the user's input.'''
        
        if str in self.all_tickers:
            self.reprompt_ticker = 'no'
        elif str not in self.all_tickers:
            matching = [s for s in self.all_tickers if str in s]
            print("\nOops! This ticker does not appear to exist.\n")
            if len(matching) > 0:
                if len(matching) == 1: print("Did you maybe mean this one: \n")
                else: print("Did you maybe mean one of these: \n")
                for tix in matching:
                    print(" - ", tix, end='')
                print('\n')
            if self.count == 2: print("HINT: Watch for special characters such as '^' or '.' and be sure to use caps when appropriate.\n")
            if self.count == 3: print("HINT: Double check that you are using the proper ticker by refering to this website: 'https://finance.yahoo.com/lookup/'\n")
            print("Please try again. ")
        return
    
    def create_data_vis(self):
        
        '''prints graphs as well as overview of stock data'''