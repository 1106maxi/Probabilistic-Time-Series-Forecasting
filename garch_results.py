#Code used to produce the final results
from src.models.garch import garch
import pandas as pd

# Start and end dates of the three study periods
start_dates =["2009-01-01", "2012-05-02", "2015-09-01"]
end_dates = ["2012-05-02", "2015-09-01", "2019-01-01"]

industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]

all_results_garch = []
for i in range(len(start_dates)):
    for stock in industrials_stocks:
            jnj_GARCH = garch(stock, start_date=start_dates[i],end_date= end_dates[i])
            jnj_GARCH.read_data()
            jnj_GARCH.garch()
            jnj_GARCH.predict(horizon = 1)
            result = jnj_GARCH.metrics()
            all_results_garch.append(result)

    
df_results_garch = pd.DataFrame(all_results_garch)
df_results_garch.to_csv("GARCH_res_comp.csv", index=False)
print(df_results_garch)