#Code used to produce the final results

#######################################################################################
# Please be aware that running this code may yield results different from those       #
# reported in my thesis. This difference is due to stochastic weight initialization   #
# in LSTMs. Results can therfore vary with each run.                                  #
#######################################################################################

from bachelorthesis import qlstm
import pandas as pd

# Start and end dates of the three study periods
start_dates =["2009-01-01", "2012-05-02", "2015-09-01"]
end_dates = ["2012-05-02", "2015-09-01", "2019-01-01"]

industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]

results = []
for rep in range(9):
    for i in range(len(start_dates)):
        for stock in industrials_stocks:
                    QLSTM_opt = qlstm(stock,start_date=start_dates[i],end_date= end_dates[i],time_steps =15, batch_size = 32,forecasting_window = 10)
                    QLSTM_opt.rolling_forecast_v(loss= QLSTM_opt.sym_loss, model = QLSTM_opt.lstm_model_mid)
                    result = QLSTM_opt.predict_eval()
                    results.append({
                        "interval_width_sum":result["interval_width_sum"],
                        "within_interval":result["within_interval"],
                        "LSTM PINAW": result["LSTM PINAW"],
                        "LSTM PICP": result["LSTM PICP"],
                        "CWC": result["LSTM CWC"],
                        "Stock": QLSTM_opt.ticker,
                        "Study Period": i + 1
                    })

df_results = pd.DataFrame(results)

df_results.to_csv("final_results_QLSTM.csv", index=False)
print(df_results)