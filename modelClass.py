from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Model:
    
    def __init__(self):
        self.predictors = ["Open", "High", "Low", "Close", "Volume"]
    
    def Get_Train_Test(self, data, a, b, c, d):
        
        ''' inputs
            data: pandas db of model data, 
            a: start row num of train set
            b: end row num of train set
            c: start row num of test set
            d: end row num of test set
            outputs
            train, test = train and test pandas df '''
        
        train = data.iloc[a:b]
        test = data.iloc[c:d]
        return train, test
        
    def RFCmodel(self):
        
        '''initializes Random Forest Classifier Model'''
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        return model

    def predict(self, train, test, predictors, model):
        
        '''predict: trains the model, uses the model to make predictions (these
        predictions are placed into a pandas Series for compatability), ensuring 
        that the indexes are consistent. Then the model predictions are concatenated with 
        the actual outcomes in the data and then this may be used for teting and 
        visualization of model effectiveness.'''
        
        model.fit(train[predictors], train["Target"])
        predictions = model.predict(test[predictors])
        predictions_Series = pd.Series(predictions, index = test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions_Series],axis=1)
        return combined
    
    def backtest(self, data, model, predictors, start=2500, step=250):
        
        '''backtest: takes in desired data, the machine learning model, the predictors as
        well as the starting number of days worth of data to train the model on. Given there 
        are typically 250 trading days/year, the function will start by training the model on
        10 years of data (2500 days) and incrementally add 1 more year of training data after
        every iteration to be used for predicting the values in the following year.'''
        
        all_predictions = []
    
        for i in range(start, data.shape[0], step):
            train, test = self.Get_Train_Test(data, 0, i, i, i+step)
            predictions = self.predict(train, test, predictors, model)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)