import numpy as np
import utils
from TimeGPTWrapper import TimeGPTWrapper
from Hmm import Hmm

def iterate_n_components(max_n_components, sequences):
    results = {}

    for curr_n_components in range(3, max_n_components + 1):
        print(f"\n=== HMM with {curr_n_components} components ===")
        # Initialize KMeans on log returns
        means_init, covars_init, X = utils.initialize_with_kmeans(sequences, curr_n_components)

        # Train HMM on log returns
        hmm_wrapper = Hmm(n_components=curr_n_components)
        hmm_wrapper.init_hmm_with_kmeans(means_init, covars_init)
        hmm_wrapper.model.fit(X)
        hmm_wrapper.plot_hmm_log_likelihood()

        # Generate log return sequences from HMM
        n_sequences = generated_sequences.shape[0]
        n_steps = generated_sequences.shape[1] - 1
        hmm_generated_log_returns = hmm_wrapper.generate_log_return_sequences(n_steps, n_sequences)

        # Compare return distributions directly
        kl_scores = []
        wasserstein_scores = []
        dtw_scores = []

        for t_ret, h_ret in zip(sequences, hmm_generated_log_returns):
            kl_scores.append(utils.compute_kl_divergence(t_ret, h_ret))
            wasserstein_scores.append(utils.compute_wasserstein_distance(t_ret, h_ret))
            dtw_scores.append(utils.compute_dtw_distance(t_ret, h_ret))

        print(f"KL Divergence Mean: {np.mean(kl_scores):.4f} ± {np.std(kl_scores):.4f}")
        print(f"Wasserstein Mean: {np.mean(wasserstein_scores):.4f} ± {np.std(wasserstein_scores):.4f}")
        print(f"DTW Distance Mean: {np.mean(dtw_scores):.4f} ± {np.std(dtw_scores):.4f}")

        results[curr_n_components] = {
            'kl_mean': np.mean(kl_scores),
            'kl_std': np.std(kl_scores),
            'w_mean': np.mean(wasserstein_scores),
            'w_std': np.std(wasserstein_scores),
            'dtw_mean': np.mean(dtw_scores),
            'dtw_std': np.std(dtw_scores)
        }

    utils.plot_metric_trends(results)

# Download historical prices
df = utils.download_series_data_yfinance()

# Generate multiple price sequences (TimeGPT)
time_gpt_wrapper = TimeGPTWrapper()
generated_sequences = time_gpt_wrapper.generate_multiple_forecasts(df, 300)

# Convert prices to log returns for HMM
log_return_sequences = utils.prices_to_log_returns(generated_sequences)

iterate_n_components(7, log_return_sequences)