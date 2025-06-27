# Install packages if needed
#import pip
#pip.main(["install", "yfinance"])
#pip.main(["install", "numpy"])
#pip.main(["install", "pandas"])
#pip.main(["install", "scikit-learn"])
#pip.main(["install", "keras"])
#pip.main(["install", "tensorflow"])
#pip.main(["install", "matplotlib"])
#pip.main(["install", "seaborn"])

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..data.data_preparation import DataPreparation



class qlstm:
    def __init__(self,ticker,forecasting_window, batch_size = 16 ,time_steps = 10,start_date = "2009-01-01",end_date = "2019-02-01"):

        #Data Collection
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize data preparation
        self.data_prep = DataPreparation(ticker, start_date, end_date)

        # Data Preparation
        self.daily_returns_scaled = None
        self.daily_prices = None
        self.daily_returns = None
        self.daily_volume = None
        self.daily_returns_sqr = None
        self.daily_volume_scaled = None
        self.combined_features = None
        ###################
        self.time_steps = time_steps
        ###################
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.x_returns = None
        self.x_volume = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_val = None
        self.y_train = None
        self.y_test = None
        self.feature_count = None
        self.window = forecasting_window
        
        # Model Training
        self.confidence = 0.9
        self.model = None
        self.epochs = 100
        self.batch_size = batch_size
        self.patience = 5
        self.sym_loss = [lambda y,f: self.q_loss((1-self.confidence)/2,y,f), lambda y,f: self.q_loss(self.confidence +((1-self.confidence)/2) ,y,f)]


        # Model Testing
        self.combined_predictions = None
        self.all_predictions = None
        self.number_of_steps = None
        self.y_test_inv = None
        self.pred_inv = None
        self. eta = 30 #Weighting factor for penalization in the CWC
        
        # Graphs 
        self.daily_dates = None
        self.dates_test = None

    # Function for creating sequential input data
    def create_lstm_data(self, data_set_scaled,num_features ,time_steps=1):
        # Use data preparation class method
        return self.data_prep.create_lstm_sequences(data_set_scaled, num_features, time_steps)
        
    # Define quantile loss function
    def q_loss(self, q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)    # Function for downloading the data if only the daily returns are used as a feature:
    def read_data_return(self):
        # Use data preparation class
        self.data_prep.download_data()
        self.data_prep.calculate_returns()
        
        # Get processed data using data preparation class
        scaled_df, self.feature_count, self.train_size, self.val_size, self.test_size = self.data_prep.prepare_data_for_qlstm(['returns'])
        
        # Create sequential data for the QLSTM
        self.x, self.y = self.data_prep.create_lstm_sequences(scaled_df, self.feature_count, self.time_steps)

        # Update other attributes for compatibility
        self.daily_prices = self.data_prep.daily_prices
        self.daily_returns = self.data_prep.daily_returns     # Function for downloading the data if the daily returns and the daily trading volume are used as features:
    def read_data_return_vol(self):
        # Use data preparation class
        self.data_prep.download_data()
        self.data_prep.calculate_returns()
        
        # Get processed data using data preparation class with returns and volume
        scaled_df, self.feature_count, self.train_size, self.val_size, self.test_size = self.data_prep.prepare_data_for_qlstm(['returns', 'volume'])
        
        # Create sequential data for the QLSTM
        self.x, self.y = self.data_prep.create_lstm_sequences(scaled_df, self.feature_count, self.time_steps)

        # Update other attributes for compatibility
        self.daily_prices = self.data_prep.daily_prices
        self.daily_returns = self.data_prep.daily_returns
        self.volume = self.data_prep.volume
        self.daily_returns_sqr = self.data_prep.daily_returns_sqr

    # The three model configurations that are tested:
    def lstm_model_complex(self, input_shape):
        # Input layer
        inputs = Input(shape=input_shape) 
        # Three LSTM layers with 100 neurons each and a dropout rate of 0.1
        lstm1 = LSTM(units=100, return_sequences=True,dropout= 0.1)(inputs)
        lstm2 = LSTM(units=100, return_sequences=True,dropout= 0.1)(lstm1)
        lstm3 = LSTM(units=100,dropout= 0.1)(lstm2)

        # Output layers for quantiles
        out10 = Dense(1)(lstm3)
        out90 = Dense(1)(lstm3)

        model = Model(inputs, [out10, out90]) 
        return model


    def lstm_model_mid(self, input_shape):
        # Input layer
        inputs = Input(shape=input_shape)
        # Two LSTM layers with 80 neurons each and a dropout rate of 0.1
        lstm1 = LSTM(units=80, return_sequences=True,dropout= 0.1)(inputs)
        lstm2 = LSTM(units=80,dropout= 0.1)(lstm1)

        # Output layers for quantiles
        out10 = Dense(1)(lstm2)
        out90 = Dense(1)(lstm2)

        model = Model(inputs, [out10, out90])
        return model
    
    def lstm_model_simpler(self, input_shape):
        # Input layer
        inputs = Input(shape=input_shape)
        # One LSTM layers with 25 neurons and a dropout rate of 0.1
        lstm1 = LSTM(units=25,dropout= 0.1)(inputs)

        # Output layers for quantiles
        out10 = Dense(1)(lstm1)
        out90 = Dense(1)(lstm1)

        model = Model(inputs, [out10, out90])
        return model

    # Function for updating the training data when the forecasting window is shifted
    def update_data_splits(self, i):
        roll = i * self.window

        self.x_train, self.y_train = self.x[:self.train_size+roll], self.y[:self.train_size+roll]
        self.x_val, self.y_val = self.x[self.train_size+roll:self.train_size + self.val_size+roll], self.y[self.train_size+roll:self.train_size + self.val_size+roll]
        self.x_test, self.y_test = self.x[self.train_size + self.val_size+roll:self.train_size + self.val_size+roll+self.window], self.y[self.train_size + self.val_size +roll:self.train_size + self.val_size+roll+self.window]

        if not isinstance(self.y_train, list) or len(self.y_train) != 2:
            self.y_train = [self.y_train, self.y_train] 
        if not isinstance(self.y_test, list) or len(self.y_test) != 2:
            self.y_test = [self.y_test, self.y_test] 

    # Forecasting function if only the daily returns are used as a feature
    def rolling_forecast(self, loss, model):
        # Read and prepare data
        self.read_data_return()

        input_shape = (self.time_steps, self.feature_count)
        self.model = model(input_shape) # The medium model is used
        
        # Compile model with quantile losses
        losses = loss
        self.model.compile(loss=losses, optimizer="adam", loss_weights=[0.5, 0.5])

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5) # early stopping 
        self.number_of_steps = self.test_size // self.window

        # Initialize container for combined predictions
        self.combined_predictions = [np.empty((0, 1)), np.empty((0, 1))]

        for step in range(self.number_of_steps):
            # Update the data splits for this window
            self.update_data_splits(step)

            # Train the model
            self.model.fit(self.x_train, self.y_train, 
                           validation_data=(self.x_val, self.y_val), 
                           epochs=self.epochs, batch_size=self.batch_size, 
                           callbacks=[stop_early])

            # Make predictions
            predictions = self.model.predict(self.x_test)

            # Add current predictions to the combined predictions
            self.combined_predictions[0] = np.vstack([self.combined_predictions[0], predictions[0]])
            self.combined_predictions[1] = np.vstack([self.combined_predictions[1], predictions[1]])
    
     # Forecasting function if the daily returns and the daily trading volume are used as features
    def rolling_forecast_v(self, loss,model):
        # Read and prepare data
        self.read_data_return_vol()

        input_shape = (self.time_steps, self.feature_count)
        self.model = model(input_shape) # The medium model is used
        
        # Compile model with quantile losses
        losses = loss
        self.model.compile(loss=losses, optimizer="adam", loss_weights=[0.5, 0.5])

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5) # early stopping 
        self.number_of_steps = self.test_size // self.window
        # Initialize container for combined predictions
        self.combined_predictions = [np.empty((0, 1)), np.empty((0, 1))]

        for step in range(self.number_of_steps):
            # Update the data splits for this window
            self.update_data_splits(step)

            # Train the model
            self.model.fit(self.x_train, self.y_train, 
                           validation_data=(self.x_val, self.y_val), 
                           epochs=self.epochs, batch_size=self.batch_size, 
                           callbacks=[stop_early])

            # Make predictions
            predictions = self.model.predict(self.x_test)

            # Add current predictions to the combined predictions
            self.combined_predictions[0] = np.vstack([self.combined_predictions[0], predictions[0]])
            self.combined_predictions[1] = np.vstack([self.combined_predictions[1], predictions[1]])

    # Test the QLSTM on the test set
    def predict_eval(self):
        self.y_test = self.y[self.train_size + self.val_size :self.train_size + self.val_size+self.number_of_steps*self.window]
        
        # Calculate upper and lower bound of the prediction intervals
        lower_bound = self.combined_predictions[0]  
        upper_bound = self.combined_predictions[1]  

        # Calculate the prediction interval normalized average width (PINAW)
        n = len(self.y_test)
        interval_width_sum = np.sum(upper_bound - lower_bound)
        y_range = np.max(self.y_test) - np.min(self.y_test)
        pinaw = interval_width_sum / (n * y_range)
        
        # Calculate the prediction interval coverage probability (PICP)
        within_interval = np.sum((self.y_test >= lower_bound) & (self.y_test <= upper_bound))
        picp = within_interval / n

        # Calculate Coverage Width-based Criterion (CWC)
            # Eta is the weighting factor for penalization
        cwc = (1 - pinaw) * np.exp(-self.eta * (picp - (self.confidence))**2)
        
        return {
            "batch size": self.batch_size,
            "time steps": self.time_steps,
            "ticker": self.ticker,
            "interval_width_sum":interval_width_sum,
            "within_interval":within_interval,
            "LSTM CWC": cwc,
            "LSTM PINAW": pinaw,
            "LSTM PICP": picp,
        }
    
    # Function for plotting the results 
    def graph(self):
        
        self.pred_inv = [
            self.data_prep.inverse_transform_predictions(self.combined_predictions[0]).flatten(),
            self.data_prep.inverse_transform_predictions(self.combined_predictions[1]).flatten()        ]
        # Get the actual length of predictions
        pred_length = len(self.combined_predictions[0])
        self.y_test_inv = self.daily_returns.iloc[self.train_size + self.val_size:self.train_size + self.val_size + pred_length]
        
        self.daily_dates = self.data_prep.get_date_index_for_period(self.train_size + self.val_size, self.train_size + self.val_size + pred_length)
        
        # Ensure all arrays have the same length
        min_length = min(len(self.daily_dates), len(self.y_test_inv), len(self.pred_inv[0]))
        self.daily_dates = self.daily_dates[:min_length]
        self.y_test_inv = self.y_test_inv[:min_length]
        pred_0 = self.pred_inv[0][:min_length].flatten()
        pred_1 = self.pred_inv[1][:min_length].flatten()

        plt.figure(figsize=(14, 5))
        plt.plot(self.daily_dates, self.y_test_inv, label="Actual Daily Return ", color="black")
        plt.plot(self.daily_dates, pred_0, label=f"Predicted Return (Quantile {round(((1-self.confidence)/2)*100,0)}%)", linestyle="dotted", color = "purple")
        plt.plot(self.daily_dates, pred_1, label=f"Predicted Return (Quantile {(self.confidence +((1-self.confidence)/2))*100}%)", linestyle="dotted", color = "green")

        plt.fill_between(self.daily_dates, pred_0, pred_1, where=(pred_1 >= pred_0), color="red", alpha=0.3)

        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend(loc="upper right") 
        plt.show()
        