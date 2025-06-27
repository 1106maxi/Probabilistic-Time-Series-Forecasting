"""
Evaluation script for comparing GARCH and Quantile LSTM model performance.

This script loads trained models and generates evaluation metrics and visualizations.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.garch import garch
from models.qlstm import qlstm

def evaluate_model_performance(garch_results, qlstm_results):
    """
    Evaluate and compare model performance.
    
    Args:
        garch_results (list): Results from GARCH models
        qlstm_results (list): Results from QLSTM models
    """
    
    # Convert to DataFrames for analysis
    df_garch = pd.DataFrame(garch_results)
    df_qlstm = pd.DataFrame(qlstm_results)
    
    print("=== GARCH Model Performance ===")
    print(f"Average PICP: {df_garch['PICP'].mean():.4f}")
    print(f"Average PINAW: {df_garch['PINAW'].mean():.4f}")
    print(f"Average CWC: {df_garch['CWC'].mean():.4f}")
    
    print("\n=== QLSTM Model Performance ===")
    print(f"Average PICP: {df_qlstm['LSTM PICP'].mean():.4f}")
    print(f"Average PINAW: {df_qlstm['LSTM PINAW'].mean():.4f}")
    print(f"Average CWC: {df_qlstm['CWC'].mean():.4f}")
    
    return df_garch, df_qlstm

def generate_sample_prediction(ticker="NOC", start_date="2009-01-01", end_date="2012-05-02"):
    """
    Generate a sample prediction visualization for demonstration.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for the sample
        end_date (str): End date for the sample
    """
    print(f"\nGenerating sample prediction for {ticker}...")
    
    # GARCH model
    print("Training GARCH model...")
    garch_model = garch(ticker, start_date=start_date, end_date=end_date)
    garch_model.read_data()
    garch_model.garch()
    garch_model.predict(horizon=1)
    garch_results = garch_model.metrics()
    
    print(f"GARCH Results - PICP: {garch_results['PICP']:.4f}, PINAW: {garch_results['PINAW']:.4f}")
    
    # QLSTM model
    print("Training QLSTM model...")
    qlstm_model = qlstm(ticker, start_date=start_date, end_date=end_date, 
                       time_steps=15, batch_size=32, forecasting_window=10)
    qlstm_model.rolling_forecast_v(loss=qlstm_model.sym_loss, model=qlstm_model.lstm_model_mid)
    qlstm_results = qlstm_model.predict_eval()
    
    print(f"QLSTM Results - PICP: {qlstm_results['LSTM PICP']:.4f}, PINAW: {qlstm_results['LSTM PINAW']:.4f}")
    
    # Generate visualization
    try:
        qlstm_model.graph()
        print("Visualization saved successfully!")
    except Exception as e:
        print(f"Visualization generation failed: {e}")

def load_existing_results():
    """Load existing results from CSV files if available."""
    garch_results = []
    qlstm_results = []
    
    try:
        # Try to load existing GARCH results
        if os.path.exists("GARCH_res_comp.csv"):
            df_garch = pd.read_csv("GARCH_res_comp.csv")
            garch_results = df_garch.to_dict('records')
            print(f"Loaded {len(garch_results)} GARCH results from CSV")
    except Exception as e:
        print(f"Could not load GARCH results: {e}")
    
    try:
        # Try to load existing QLSTM results
        if os.path.exists("final_results_QLSTM.csv"):
            df_qlstm = pd.read_csv("final_results_QLSTM.csv")
            qlstm_results = df_qlstm.to_dict('records')
            print(f"Loaded {len(qlstm_results)} QLSTM results from CSV")
    except Exception as e:
        print(f"Could not load QLSTM results: {e}")
    
    return garch_results, qlstm_results

if __name__ == "__main__":
    print("Starting model evaluation...")
    
    # Load existing results
    garch_results, qlstm_results = load_existing_results()
    
    if garch_results and qlstm_results:
        # Evaluate performance
        df_garch, df_qlstm = evaluate_model_performance(garch_results, qlstm_results)
        
        # Generate sample prediction
        generate_sample_prediction()
        
    else:
        print("No existing results found. Please run training first using train.py")
        print("Or run the original scripts: garch_results.py and qlstm_results.py")
    
    print("Evaluation completed!")
