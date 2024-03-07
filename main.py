
# File: main
# Author: Taylor Kramer
# Date: 03/02/24

'''Prompt: Creating and testing machine learning models to predict the direction of stocks.
   Answering the question: Will a certain stock go up or down tomorrow? How well can 
   various machine learning models make these predictions.''' 

from data import Data
from model import Model
import pandas as pd
from testing import Testing
from joblib import dump # for pickling model


def main():
   #TODO: create an interaction class to place main content
   repeat = 'yes'

   while repeat == 'yes':
   
      #--------Data-----------------------------------------------------------------------------------
      
      print("\nHowdy! :) What data are you looking to analyze today?")
      print("Here are some examples: ^GSPC (S&P 500), ^DJI (Dow Jones), AAPL (Apple), TSLA (Tesla)\n")
      ticker = input("Your choice: ")
      "TODO: import list of all options, and reprompt if choice doesn't exist, have the reprompt give tips and suggestions"
      calldata = Data(ticker) #initiates instance of Data class -> imports raw data
      calldata.clean_data() #wrangles data
      mydata = calldata.get_data() #places cleaned data into variable: mydata
      #TODO: create visualizations for dataset, create visualization class
         
      #--------Model----------------------------------------------------------------------------------
      
      print("\nGreat choice!\nNow, what type of ML model are you looking to use?")
      print("Here are the options we have available as of right now: RFC (Randon Forest Classifier)\n")
      modeltype = input("Your choice: ")
      "TODO: reprompt if choice doesn't exist, have the reprompt give tips"
      print("\nAwesome! I am initializing and training your model on your chosen dataset now...\n")
      mymodel = Model(calldata, modeltype) #initiates instance of Model class -> initializes model
      mymodel.train_model(calldata, pd.DataFrame(), pd.DataFrame()) #trains base model
      #TODO: create visualizations for the initial trained model
      print("\n All Done! Now time to do a little bit of testing.")
      
      #--------Testing & Improvement------------------------------------------------------------------

      print("\nFirst, let's do some backtesting... starting now...")
      
      # back testing model
      # providing results
      # improving model
      # testing again
      # providing results

         #--------Finish---------------------------------------------------------------------------------
      print("We are all done!\n")
      print("Would you like to save this model?\n")
      save_model = input("'yes' or 'no': ")
      if save_model == "yes":
         dump(mymodel, 'saved_model.joblib') #-> have name depend on data and type of model # pickles finalized model for use in the future
         print("\nYay! Your model has been pickled and saved into a file called: _____\n")
         print("HINT: to use the pickled model elsewhere: from joblib import load and then savedmodel = load('mymodel.joblib')\n")
      else:
         print("Okay! ")
      print("Would you like to try anything else today?\n")
      repeat = input("'yes' or 'no': ")

if __name__ == "__main__":
   main()