import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreparation:
    """
    A class for handling common data preparation tasks for financial time series analysis.
    This class provides functionality shared between GARCH and QLSTM models.
    """
    
    def __init__(self, ticker, start_date="2009-01-01", end_date="2019-02-01"):
        """
        Initialize the DataPreparation class.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Raw data
        self.data = None
        self.daily_prices = None
        self.volume = None
        
        # Processed data
        self.daily_returns = None
        self.daily_returns_scaled = None
        self.daily_returns_sqr = None
        self.abs_daily_returns = None
          # Scaler for QLSTM
        self.scaler = StandardScaler()
    
    def download_data(self):
        """
        Download stock data from Yahoo Finance.
        """
        try:
            # Try multiple approaches to download data
            # Approach 1: Standard download
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, 
                                  progress=False, threads=False)
            
            # If empty, try using Ticker object
            if self.data.empty:
                ticker_obj = yf.Ticker(self.ticker)
                self.data = ticker_obj.history(start=self.start_date, end=self.end_date)
            
            # If still empty, try with period parameter for recent data
            if self.data.empty and self.start_date >= "2020-01-01":
                self.data = yf.download(self.ticker, period='2y', progress=False)
                # Filter to requested date range
                self.data = self.data[self.start_date:self.end_date]
            
            if self.data.empty:
                raise ValueError(f"No data downloaded for ticker {self.ticker} between {self.start_date} and {self.end_date}")
                
            self.daily_prices = self.data["Close"]
            # Ensure both prices and volume are proper Series
            if hasattr(self.daily_prices, 'iloc') and self.daily_prices.ndim > 1:
                self.daily_prices = self.daily_prices.iloc[:, 0]
                
            # Ensure volume is a Series, not a DataFrame column
            volume_data = self.data["Volume"]
            if hasattr(volume_data, 'iloc') and volume_data.ndim > 1:
                self.volume = volume_data.iloc[:, 0]
            else:
                self.volume = volume_data
                
        except Exception as e:
            raise ValueError(f"Failed to download data for ticker {self.ticker}: {str(e)}")
        
    def calculate_returns(self):
        """
        Calculate daily returns and derived metrics.
        """
        if self.daily_prices is None:
            raise ValueError("Daily prices not available. Call download_data() first.")
              # Calculate daily returns
        self.daily_returns = (self.daily_prices / self.daily_prices.shift(1)) - 1
        self.daily_returns = self.daily_returns.dropna()
        
        # Calculate derived metrics
        self.daily_returns_scaled = self.daily_returns * 100  # For GARCH (percentage)
        self.daily_returns_sqr = self.daily_returns ** 2
        self.abs_daily_returns = self.daily_returns ** 2  # Same as squared for GARCH
        
    def prepare_data_for_garch(self, train_ratio=0.8):
        """
        Prepare data specifically for GARCH model.
        
        Args:
            train_ratio (float): Ratio of data to use for training
            
        Returns:
            tuple: (train_data, test_data, split_date)
        """
        if self.daily_returns_scaled is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")
            
        if len(self.daily_returns_scaled) == 0:
            raise ValueError("No data available after calculating returns.")
            
        train_size = int(len(self.daily_returns_scaled) * train_ratio)
        
        if train_size == 0:
            raise ValueError(f"Train size is 0. Check if sufficient data is available. Total data points: {len(self.daily_returns_scaled)}")
            
        train = self.daily_returns_scaled[:train_size]
        test = self.daily_returns_scaled[train_size:]
        split_date = train.index[-1]
        
        return train, test, split_date
        
    def prepare_data_for_qlstm(self, features=['returns'], train_ratio=0.6, val_ratio=0.2):
        """
        Prepare data for QLSTM model with proper scaling and feature selection.
        
        Args:
            features (list): List of features to include ['returns', 'volume']
            train_ratio (float): Ratio of data to use for training
            val_ratio (float): Ratio of data to use for validation
            
        Returns:
            tuple: (scaled_data, feature_count, train_size, val_size, test_size)
        """
        if self.daily_returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")
            
        # Prepare feature dataframe
        df_dict = {}
        
        if 'returns' in features:
            df_dict['Returns'] = self.daily_returns
            
        if 'volume' in features:
            if self.volume is None:
                raise ValueError("Volume data not available. Call download_data() first.")
            # Align volume data with returns and ensure it's a Series
            aligned_volume = self.volume.reindex(self.daily_returns.index)
            # Ensure it's really a 1D Series
            if hasattr(aligned_volume, 'values'):
                aligned_volume = pd.Series(aligned_volume.values.flatten(), 
                                         index=self.daily_returns.index, name='Volume')
            df_dict['Volume'] = aligned_volume
            
        df = pd.DataFrame(df_dict, index=self.daily_returns.index)
        feature_count = len(df.columns)
        
        # Scale the data
        scaled_df = self.scaler.fit_transform(df)
        
        # Calculate split sizes
        total_size = len(scaled_df)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        return scaled_df, feature_count, train_size, val_size, test_size
        
    def create_lstm_sequences(self, scaled_data, feature_count, time_steps=10):
        """
        Create sequential input data for LSTM models.
        
        Args:
            scaled_data (numpy.ndarray): Scaled feature data
            feature_count (int): Number of features
            time_steps (int): Number of time steps for sequences
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target
        """
        X = []
        
        for j in range(feature_count):
            X.append([])
            for i in range(time_steps, scaled_data.shape[0]):
                X[j].append(scaled_data[i-time_steps:i, j])
                
        X = np.moveaxis(X, [0], [2])
        X, yi = np.array(X), np.array(scaled_data[time_steps:, 0])
        y = np.reshape(yi, (len(yi), 1))
        
        return X, y
        
    def get_date_index_for_period(self, start_idx, end_idx):
        """
        Get date index for a specific period.
        
        Args:
            start_idx (int): Start index
            end_idx (int): End index
            
        Returns:
            pandas.DatetimeIndex: Date index for the specified period
        """
        if self.daily_returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")
            
        return self.daily_returns.index[start_idx:end_idx]
        
    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform scaled predictions back to original scale.
        
        Args:
            predictions (numpy.ndarray): Scaled predictions
            
        Returns:
            numpy.ndarray: Predictions in original scale
        """
        return self.scaler.inverse_transform(predictions.reshape(-1, 1))
