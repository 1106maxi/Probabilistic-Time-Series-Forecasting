# Code used for plotting the daily returns, squared daily returns and their respective autocorrelation function as well as conducting the ADF test, Ljung-box test and calculating the kurtosis:
from bachelorthesis import garch
import pandas as pd


industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]

all_results_garch = []
for stock in industrials_stocks:
    GARCH_model = garch(stock, start_date="2009-01-01", end_date="2019-01-01")
    GARCH_model.read_data()
    
    stationarity_result = GARCH_model.stationarity_test()
    acf_result = GARCH_model.acf_test()
    GARCH_model.kurtosis()
    
    combined_result = pd.concat([stationarity_result, acf_result.drop(columns="ticker")], axis=1)
    
    all_results_garch.append(combined_result)

    GARCH_model.plot_acf()
    GARCH_model.stationarity()


df_results_garch = pd.concat(all_results_garch, ignore_index=True)
print(df_results_garch)
