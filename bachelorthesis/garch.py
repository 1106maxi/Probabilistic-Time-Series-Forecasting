# Install packages if needed
#import pip
#pip.main(["install", "yfinance"])
#pip.main(["install", "statsmodels"])
#pip.main(["install", "arch"])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
from scipy.stats import norm, kurtosis
from arch.unitroot import ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from .data_preparation import DataPreparation

class garch():
    def __init__ (self,ticker,start_date,end_date):
        
        #Data Collection
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize data preparation
        self.data_prep = DataPreparation(ticker, start_date, end_date)

        # Data Preparation
        self.daily_prices = None
        self.daily_returns = None
        self.daily_returns_scaled = None
        self.abs_daily_returns = None
        self.train_size = None
        self.train = None
        self.test = None
        
        # Model Training
        self.confidence = 0.9
        self.model = None
        self.model_fit = None
        self.order_parameters = []
        self.garch_aic = []
        
        # Model Prediction
        self.split_date = None
        self.forecasts = None
        self.test_decimal = None
        self.mean_forecast_decimal = None
        self.lower_bound = None
        self.upper_bound = None

        # Model Evaluation
        self.eta = 30 #Weighting factor for penalization in the CWC

    def read_data(self):
        # Use data preparation class
        self.data_prep.download_data()
        self.data_prep.calculate_returns()
        
        # Get processed data
        self.daily_prices = self.data_prep.daily_prices
        self.daily_returns = self.data_prep.daily_returns
        self.daily_returns_scaled = self.data_prep.daily_returns_scaled
        self.abs_daily_returns = self.data_prep.abs_daily_returns
    
        # Split the data into training and test set 
        self.train, self.test, self.split_date = self.data_prep.prepare_data_for_garch()

    def stationarity(self):
        # Plot Daily Returns
        plt.figure(figsize=(10, 5))
        plt.plot(self.daily_returns, color="black", lw=1, label="Daily Returns")
        plt.title("Daily Returns", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Returns", fontsize=14)

        plt.legend()
        plt.grid(visible=False)
        plt.ylim(-0.115, 0.115) 
        plt.tight_layout()
        plt.show()
       
        # Plot Squared Daily Returns
        plt.figure(figsize=(10, 5))
        plt.plot(self.daily_returns ** 2, color="black", lw=1, label="Squared Daily Returns")
        plt.title("Squared Daily Returns", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Squared Returns", fontsize=14)

        plt.legend()
        plt.ylim(-0.0001, 0.015)
        plt.grid(visible=False)
        plt.tight_layout()

        plt.show()

    # Plotting autocorrelation functions
    def plot_acf(self):
        plt.figure(figsize=(14, 14))

        plt.subplot(3, 1, 1)
        plot_acf(self.daily_returns, ax=plt.gca(), title="ACF of Daily Returns", color="blue")
        plt.title("ACF of Daily Returns", fontsize=16)
        plt.xlabel("Lags", fontsize=14)
        plt.ylabel("ACF", fontsize=14)
        plt.grid(visible=False)

        plt.subplot(3, 1, 2)
        plot_acf(self.abs_daily_returns, ax=plt.gca(), title="ACF of Squared Daily Returns", color="red")
        plt.title("ACF of Squared Daily Returns", fontsize=16)
        plt.xlabel("Lags", fontsize=14)
        plt.ylabel("ACF", fontsize=14)
        plt.grid(visible=False)

        plt.tight_layout()
        plt.show()


    # ADF test
    def stationarity_test(self):
        adf = ADF(self.daily_returns, lags = 30)
        result = pd.DataFrame({"ticker": [self.ticker], 
                               "adf_statistic": [adf.stat], 
                               "p_value": [adf.pvalue]})
        return result


    # Ljung-Box test for daily returns and squared daily returns
    def acf_test(self):
        ljung_box_test = acorr_ljungbox(self.daily_returns, lags=[30], return_df=True)
        ljung_box_test_abs = acorr_ljungbox(self.abs_daily_returns, lags=[30], return_df=True)
        
        result = pd.DataFrame({
            "ticker": [self.ticker],
            "ljung_box_stat_daily": [ljung_box_test["lb_stat"].iloc[0]],
            "p_value_daily": [ljung_box_test["lb_pvalue"].iloc[0]],
            "ljung_box_stat_abs": [ljung_box_test_abs["lb_stat"].iloc[0]],
            "p_value_abs": [ljung_box_test_abs["lb_pvalue"].iloc[0]],
        })
        return result

    
    def kurtosis(self):
        kurt = kurtosis(self.daily_returns, fisher=True)
        print(f"Kurtosis of {self.ticker}: {kurt}")

    # Selecting the order parameters via AIC and fitting the model to the data
    def garch(self): 
        # Computing the AIC for every possible combination of order parameters
        for i in range(1,3):
            for j in range(3):
                self.order_parameters += [[i,j]] 
                model = arch_model(self.train,p=i,q=j).fit(disp=False) # (not) showing optimization process
                self.garch_aic += [model.aic]
        
        # Selecting best model according to AIC and fitting on the training set
        self.model = arch_model(self.daily_returns_scaled, p=self.order_parameters[np.argmin(self.garch_aic)][0],q=self.order_parameters[np.argmin(self.garch_aic)][1], dist = "normal")
        self.model_fit = self.model.fit(disp=False, last_obs=self.split_date)
    
    def predict(self, horizon=1):
        # Computing 1-step ahead predictions
        self.forecasts = self.model_fit.forecast(horizon=horizon, start=self.split_date)
        variance_forecast = self.forecasts.variance[self.split_date:].iloc[:, 0]
        mean_forecast = self.forecasts.mean[self.split_date:].iloc[:, 0]

        # Adjusting output
        self.test_decimal = self.test * 0.01
        self.mean_forecast_decimal = mean_forecast * 0.01

        # Calculate upper and lower bound of the prediction interval
        self.lower_bound = self.mean_forecast_decimal - norm.ppf(1-((1-self.confidence)/2)) * np.sqrt(variance_forecast) * 0.01
        self.upper_bound = self.mean_forecast_decimal + norm.ppf(1-((1-self.confidence)/2)) * np.sqrt(variance_forecast) * 0.01
    
        # Assuring matching indices and consistent data types
        self.test_decimal = self.test_decimal.reindex(self.mean_forecast_decimal.index)
        self.lower_bound = self.lower_bound.reindex(self.mean_forecast_decimal.index)
        self.upper_bound = self.upper_bound.reindex(self.mean_forecast_decimal.index)
        
        # Convert test_decimal to Series if it's a DataFrame to ensure compatibility
        if hasattr(self.test_decimal, 'shape') and len(self.test_decimal.shape) > 1:
            self.test_decimal = self.test_decimal.iloc[:, 0]

    # Compute metrics
    def metrics(self):
        # Compute the prediction interval normalized average width (PINAW)
        n = len(self.test_decimal)
        interval_width_sum = np.sum(self.upper_bound - self.lower_bound)
        y_range = np.max(self.test_decimal) - np.min(self.test_decimal)
        pinaw = interval_width_sum / (n * y_range)

        # Compute the prediction interval coverage probability (PICP)
        within_interval = np.sum((self.test_decimal >= self.lower_bound) & (self.test_decimal <= self.upper_bound))
        picp = within_interval / n

        # Calculate Coverage Width-based Criterion (CWC)
            # Eta is the weighting factor for penalization
        cwc = (1 - pinaw) * np.exp(-self.eta * (picp - (self.confidence))**2)

        return {
            "ticker": self.ticker,
            "interval_width_sum": interval_width_sum,
            "within_interval": within_interval,
            "CWC": cwc,
            "PINAW": pinaw,
            "PICP": picp,
        }
    
    # Plotting results
    def graph(self):  
        plt.figure(figsize=(14, 5))
        plt.plot(self.test_decimal.index, self.test_decimal, label="Actual Daily Return", color="black")
        plt.fill_between(self.mean_forecast_decimal.index, self.lower_bound, self.upper_bound,
                         color="red", alpha=0.3, label=f"{self.confidence*100}% Prediction Interval") # alpha means transparency
        plt.xlabel("Date")
        plt.ylabel("Daily Returns")
        plt.legend()
        plt.show()