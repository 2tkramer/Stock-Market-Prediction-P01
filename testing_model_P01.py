
# File: testing_model_P02
# Author: Taylor Kramer
# Date: 03/02/24

# this file is dedicated to testing the model created in file: model_P02.py

#%%

#importing neccessary libraries
from model_P02 import test, predictors
import pandas as pd
import sklearn
from joblib import load
from sklearn.metrics import precision_score

#importing model from file

model = load('saved_model.joblib')

#--------Testing Model------------------------------------------------------------

'''Use the model to predict the targets in the test set of the data. Utilize 
sklearn's precision_score function to determine what percentage of the time the 
model correctly predicts what direction the market will go.'''

predictions = model.predict(test[predictors])
#print(predictions)
score = precision_score(test["Target"], predictions)
print(score)

'''Begin backtesting model...'''



#--------Visualizing Outcomes-----------------------------------------------------

'''Starts by placing predictions into a pandas Series for compatibility, ensuring 
that the indexes are consistent. Then the model predictions are concatenated with 
the actual outcomes as seen in the data and then this may be plotted to visualize 
the effectiveness of the model.'''

predictions_Series = pd.Series(predictions, index = test.index)
combined = pd.concat([test["Target"], predictions_Series],axis=1)
combined.plot()

# %%
