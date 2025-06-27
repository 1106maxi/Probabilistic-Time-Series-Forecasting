#!/usr/bin/env python3
"""
Training entry script for probabilistic time series forecasting models.

This script trains both GARCH and Quantile LSTM models on financial time series data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.garch import garch
from models.qlstm import qlstm

def train_garch_models():
    """Train GARCH models on the specified datasets."""
    print("Training GARCH models...")
    
    # Start and end dates of the three study periods
    start_dates = ["2009-01-01", "2012-05-02", "2015-09-01"]
    end_dates = ["2012-05-02", "2015-09-01", "2019-01-01"]
    
    industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]
    
    all_results_garch = []
    for i in range(len(start_dates)):
        for stock in industrials_stocks:
            print(f"Training GARCH for {stock} ({start_dates[i]} to {end_dates[i]})")
            
            garch_model = garch(stock, start_date=start_dates[i], end_date=end_dates[i])
            garch_model.read_data()
            garch_model.garch()
            garch_model.predict(horizon=1)
            result = garch_model.metrics()
            all_results_garch.append(result)
    
    return all_results_garch

def train_qlstm_models():
    """Train Quantile LSTM models on the specified datasets."""
    print("Training Quantile LSTM models...")
    
    # Start and end dates of the three study periods
    start_dates = ["2009-01-01", "2012-05-02", "2015-09-01"]
    end_dates = ["2012-05-02", "2015-09-01", "2019-01-01"]
    
    industrials_stocks = ["BA", "CHRW", "DOV", "EFX", "EMR", "FAST", "ITW", "NOC", "UPS", "WM"]
    
    results = []
    for rep in range(3):  # Reduced repetitions for demo
        for i in range(len(start_dates)):
            for stock in industrials_stocks:
                print(f"Training QLSTM for {stock} ({start_dates[i]} to {end_dates[i]}) - Rep {rep+1}")
                
                qlstm_model = qlstm(stock, start_date=start_dates[i], end_date=end_dates[i], 
                                   time_steps=15, batch_size=32, forecasting_window=10)
                qlstm_model.rolling_forecast_v(loss=qlstm_model.sym_loss, model=qlstm_model.lstm_model_mid)
                result = qlstm_model.predict_eval()
                results.append({
                    "interval_width_sum": result["interval_width_sum"],
                    "within_interval": result["within_interval"],
                    "LSTM PINAW": result["LSTM PINAW"],
                    "LSTM PICP": result["LSTM PICP"],
                    "CWC": result["LSTM CWC"],
                    "ticker": result["ticker"],
                    "batch size": result["batch size"],
                    "time steps": result["time steps"]
                })
    
    return results

if __name__ == "__main__":
    print("Starting model training...")
    
    # Train GARCH models
    garch_results = train_garch_models()
    print(f"GARCH training completed. {len(garch_results)} models trained.")
    
    # Train QLSTM models
    qlstm_results = train_qlstm_models()
    print(f"QLSTM training completed. {len(qlstm_results)} models trained.")
    
    print("All training completed successfully!")
