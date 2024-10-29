from bachelorthesis import *

GARCH_test = garch("NOC", start_date="2009-01-01", end_date="2012-05-02")
GARCH_test.read_data()
GARCH_test.garch()
GARCH_test.predict(horizon=1)
GARCH_test.graph()

QLSTM_test = qlstm("BA",start_date="2012-05-03",end_date= "2015-09-01",time_steps =15, batch_size = 32,forecasting_window = 10)
QLSTM_test.rolling_forecast(loss= QLSTM_test.sym_loss,model = QLSTM_test.lstm_model_mid)
QLSTM_test.predict_eval()
QLSTM_test.graph()

