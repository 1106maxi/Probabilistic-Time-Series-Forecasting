#######################################################################################
# Please be aware that running this code may yield results different from those       #
# reported in my thesis. This difference is due to stochastic weight initialization   #
# in LSTMs. Results can therfore vary with each run.                                  #
#######################################################################################

from bachelorthesis import qlstm
import pandas as pd

# Determining the start of the test set for each period so it can be excluded from the hyperparameter tuning:

# The start and end dates of the three study periods
start_dates = ["2009-01-01", "2012-05-02", "2015-09-01"]
end_dates = ["2012-05-02", "2015-09-01", "2019-01-01"]

# Loop through each pair of start and end dates
for i,(start, end) in enumerate(zip(start_dates, end_dates)):

    # Convert the start and end dates to datetime format
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Calculate the total duration between the start and end date
    total_duration = end_date - start_date
    
    # The test period will be the last 20% of the total period
    test_duration = total_duration * 0.2
    
    # The start of the test period will be at 80% of the total period
    test_date = end_date - test_duration
    
    # Print the start of the test period
    print(f"Start of test period {i+1}: {test_date.date()}")



# Code used for testing the three different QLSTM architectures:

time_steps = [5, 10, 15, 20, 25, 30]
batch_size = [16, 32, 64, 128]

# United Parcel Service for the first study period
results_ups_simp = []
for rep in range(1):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("UPS",start_date="2009-01-01",end_date= "2011-09-01",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_simpler)
                    result = QLSTM_opt.predict_eval()
                    results_ups_simp.append({
                        "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_ups_simp = pd.DataFrame(results_ups_simp)
df_results_ups_simp.to_csv("ups_simple.csv", index=False)
print(df_results_ups_simp)


results_ups_mid = []
for rep in range(9):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("UPS",start_date="2009-01-01",end_date= "2011-09-01",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_mid)
                    result = QLSTM_opt.predict_eval()
                    results_ups_mid.append({
                        "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_ups_mid = pd.DataFrame(results_ups_mid)
df_results_ups_mid.to_csv("ups_medium.csv", index=False)
print(df_results_ups_mid)


results_ups_comp = []
for rep in range(9):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("UPS",start_date="2009-01-01",end_date= "2011-09-01",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_complex)
                    result = QLSTM_opt.predict_eval()
                    results_ups_comp.append({
                        "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_ups_comp = pd.DataFrame(results_ups_comp)
df_results_ups_comp.to_csv("ups_complex.csv", index=False)
print(df_results_ups_comp)

# Waste Management for the second study period
results_wm_simp = []
for rep in range(9):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("WM",start_date="2012-05-02",end_date= "2014-12-31",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_simpler)
                    result = QLSTM_opt.predict_eval()
                    results_ups_simp.append({
                        "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_wm_simp = pd.DataFrame(results_wm_simp)
df_results_wm_simp.to_csv("wm_simple.csv", index=False)
print(df_results_wm_simp)

results_wm_mid = []
for rep in range(9):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("WM",start_date="2012-05-02",end_date= "2014-12-31",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_mid)
                    result = QLSTM_opt.predict_eval()
                    results_ups_mid.append({
                       "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_wm_mid = pd.DataFrame(results_wm_mid)
df_results_wm_mid.to_csv("wm_medium.csv", index=False)
print(df_results_wm_mid)

results_wm_comp = []
for rep in range(9):
    for time in time_steps:
        for batch in batch_size:
                    QLSTM_opt = qlstm("WM",start_date="2012-05-02",end_date= "2014-12-31",time_steps =time, batch_size = batch,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model= QLSTM_opt.lstm_model_complex)
                    result = QLSTM_opt.predict_eval()
                    results_ups_comp.append({
                        "time steps": result["time steps"],
                        "batch size": result["batch size"],
                        "CWC": result["LSTM CWC"]
                    })

df_results_wm_comp = pd.DataFrame(results_wm_comp)
df_results_wm_comp.to_csv("wm_complex.csv", index=False)
print(df_results_wm_comp)


#Code used for testing different features:

# Start and end dates for the three study periods without their test sets
start_dates =["2009-01-01", "2012-05-02", "2015-09-01"]
end_dates = ["2011-09-01", "2014-12-31", "2018-05-02"]

industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]

results_return = []
for rep in range(9):
    for i in range(len(start_dates)):
        for stock in industrials_stocks:
                    QLSTM_opt = qlstm(stock,start_date=start_dates[i],end_date= end_dates[i],time_steps =15, batch_size = 32,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast(loss= QLSTM_opt.sym_loss, model = QLSTM_opt.lstm_model_mid)
                    result = QLSTM_opt.predict_eval()
                    results_return.append({
                        "interval_width_sum":result["interval_width_sum"],
                        "within_interval":result["within_interval"],
                        "PINAW": result["LSTM PINAW"],
                        "PICP": result["LSTM PICP"],
                        "CWC": result["LSTM CWC"],
                        "Stock": QLSTM_opt.ticker,
                        "Study Period": i + 1
                    })

    df_return = pd.DataFrame(results_return)
        
    file_name = f"f_return_p{i+1}.csv"  
    df_return.to_csv(file_name, index=False)
    print(df_return)



results_return_vol = []
for rep in range(9):
    for i in range(len(start_dates)):
        for stock in industrials_stocks:
                    QLSTM_opt = qlstm(stock,start_date=start_dates[i],end_date= end_dates[i],time_steps =15, batch_size = 32,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast_v(loss= QLSTM_opt.sym_loss, model = QLSTM_opt.lstm_model_mid)
                    result = QLSTM_opt.predict_eval()
                    results_return_vol.append({
                        "interval_width_sum":result["interval_width_sum"],
                        "within_interval":result["within_interval"],
                        "PINAW": result["LSTM PINAW"],
                        "PICP": result["LSTM PICP"],
                        "CWC": result["LSTM CWC"],
                        "Stock": QLSTM_opt.ticker,
                        "Study Period": i + 1 
                    })


    df_return_vol = pd.DataFrame(results_return_vol)
        
    file_name = f"f_return_vol_p{i+1}.csv"  
    df_return_vol.to_csv(file_name, index=False)
    print(df_return_vol)
