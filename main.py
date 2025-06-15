import utils
from TimeGPTWrapper import TimeGPTWrapper
from Hmm import Hmm

# Initialization and generation
time_gpt_wrapper = TimeGPTWrapper()
df = utils.download_series_data_yfinance()
generated_sequences = time_gpt_wrapper.generate_multiple_forecasts(df)

# Kmeans
means_init, covars_init, X = utils.initialize_with_kmeans(generated_sequences)

# HMM
hmm_wrapper = Hmm()
hmm_wrapper.model.fit(X)

