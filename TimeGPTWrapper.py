import numpy as np
from nixtlats import NixtlaClient
import utils

class TimeGPTWrapper:
    def __init__(self):
        self.client = NixtlaClient(api_key=utils.read_api_key())

    def generate_multiple_forecasts(self, df, num_samples=10, h=100):
        forecasts = []
        for i in range(num_samples):
            df_perturbed = df.copy()
            df_perturbed['y'] += np.random.normal(0, 0.5, size=len(df))  # small noise
            forecast = self.client.forecast(df_perturbed, h=h, freq='B')
            forecast = forecast.rename(columns={'TimeGPT': 'y'})
            forecasts.append(forecast['y'].values)
        return np.array(forecasts)

