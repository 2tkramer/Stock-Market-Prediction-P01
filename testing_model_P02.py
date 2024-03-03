
# File: testing_model_P02
# Author: Taylor Kramer
# Date: 03/02/24

# this file is dedicated to testing the model created in file: model_P02.py

#importing neccessary libraries
import model_P02
from model_P02 import test, predictors
import pandas
import sklearn
from joblib import load
from sklearn.metrics import precision_score

#importing model from file

model = load('saved_model.joblib')

#--------Testing Model------------------------------------------------------------

'''Utilize sklearn's precision score to determine what percentage of the time the 
model correctly predicts what direction the market will go'''

predictions = model.predict(test[predictors])
print(predictions)
score = precision_score(test["Target"], predictions)
print(score)