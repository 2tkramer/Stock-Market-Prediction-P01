# File: main
# Author: Taylor Kramer
# Date: 03/02/24

'''Prompt: Creating and testing machine learning models to predict the direction of stocks.
   Answering the question: Will a certain stock go up or down tomorrow? How well can 
   various machine learning models make these predictions.''' 

from data import Data
from model import Model
#import testing
from joblib import dump


def main():
   
   #--------Data-----------------------------------------------------------------------------------
   
   print("\nHowdy! :) What data are you looking to analyze today?")
   print("Here are some examples: ^GSPC (S&P 500), ^DJI (Dow Jones), AAPL (Apple), TSLA (Tesla)\n")
   ticker = input("Your choice: ")
   "TODO: import list of all options, and reprompt if choice doesn't exist, have the reprompt give tips"
   calldata = Data(ticker) #initiates instance of Data class to import raw data
   calldata.clean_data() #wrangles data
   mydata = calldata.get_data() #places cleaned data into variable: mydata
      
   #--------Model----------------------------------------------------------------------------------
   
   print("\nGreat choice!\nNow, what type of ML model are you looking to use?")
   print("Here are the options we have available as of right now: RFC (Randon Forest Classifier)\n")
   mymodel = input("Your choice: ")
   "TODO: reprompt if choice doesn't exist, have the reprompt give tips"
   print("\nAwesome! I am initializing and training your model on your chosen dataset now...\n")
   mainmodel = Model(mydata, mymodel) #initiates instance of Model class which initializes model
   #TODO: fix mainmodel.train_model(None,None) #trains base model
   print("\n All Done! Now time to do a little bit of testing...")
   #TODO: create visualizations for the initial trained model
   
   #--------Testing--------------------------------------------------------------------------------

# back testing model
# providing results
# improving model
# testing again
# providing results
# pickles finalized model for use in the future
#dump(mymodel, 'saved_model.joblib')
# would you like to try something else?

if __name__ == "__main__":
    main()