#from utils.machine_learning.get_long_df import *
import pandas as pd

from utils.machine_learning.lstm_acsending_channel import CorridorModelTrainer

history_df = pd.read_csv('files/history_data/fulldata_current_history_data_binance_BTCUSDT.csv')
history_df = history_df[-len(history_df)//12:]
day_df = pd.read_csv('files/history_data/last1000_current_history_data_binance_BTCUSDT.csv')


#predictor = CorridorModelPredictor('namer', 'BTCUSDT')
#predictor.predict(day_df)


trainer = CorridorModelTrainer(model_name='lstm_v1', pair='BTCUSDT', model_dir='files/models')
meta = trainer.fit(history_df)
