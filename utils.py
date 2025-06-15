import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy, gaussian_kde
from scipy.stats import wasserstein_distance
from dtaidistance import dtw

def read_api_key(filepath: str="api_key.txt") -> str:
    try:
        with open(filepath) as f:
            api_key = f.readline().strip()
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found")

def download_series_data_yfinance() -> pd.DataFrame:
    data = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def prices_to_log_returns(sequences: np.ndarray) -> np.ndarray:
    """
    Convert an array of price sequences to log return sequences.
    sequences: shape (n_samples, n_steps)
    returns: shape (n_samples, n_steps-1)
    """
    return np.log(sequences[:, 1:] / sequences[:, :-1])

def initialize_with_kmeans(log_return_sequences: np.ndarray, n_states: int = 3):
    # Step 1: Convert each forecast sequence to returns
    return_seqs = [seq.reshape(-1, 1) for seq in log_return_sequences]
    X = np.vstack(return_seqs)  # shape: (total_returns, 1)

    # Step 2: KMeans clustering on returns
    kmeans = KMeans(n_clusters=n_states, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Step 3: Initialize means and covariances
    means_init = kmeans.cluster_centers_

    covars_init = []
    for i in range(n_states):
        cluster_points = X[labels == i]
        var = np.var(cluster_points) if len(cluster_points) > 1 else 1e-4
        covars_init.append([[var]])  # full covariance
    covars_init = np.array(covars_init)  # shape: (n_states, 1, 1)

    return means_init, covars_init, X


# 2. KL Divergence between return distributions (via histogram/binning)
def compute_kl_divergence(p_returns, q_returns, bins=50):
    p_hist, _ = np.histogram(p_returns, bins=bins, density=True)
    q_hist, _ = np.histogram(q_returns, bins=bins, density=True)
    p_hist += 1e-10  # avoid log(0)
    q_hist += 1e-10
    return entropy(p_hist, q_hist)  # KL(P || Q)

# 3. Wasserstein Distance (Distributional Overlap)
def compute_wasserstein_distance(p_returns, q_returns):
    return wasserstein_distance(p_returns, q_returns)

def compute_dtw_distance(p_returns: np.ndarray, q_returns: np.ndarray) -> float:
    return dtw.distance(p_returns, q_returns)

def plot_metric_trends(results_dict):
    components = sorted(results_dict.keys())

    kl_means = [results_dict[k]['kl_mean'] for k in components]
    kl_stds = [results_dict[k]['kl_std'] for k in components]

    w_means = [results_dict[k]['w_mean'] for k in components]
    w_stds = [results_dict[k]['w_std'] for k in components]

    dtw_means = [results_dict[k]['dtw_mean'] for k in components]
    dtw_stds = [results_dict[k]['dtw_std'] for k in components]

    plt.figure(figsize=(12, 6))

    plt.errorbar(components, kl_means, yerr=kl_stds, label='KL Divergence', marker='o')
    plt.errorbar(components, w_means, yerr=w_stds, label='Wasserstein Distance', marker='s')
    plt.errorbar(components, dtw_means, yerr=dtw_stds, label='DTW Distance', marker='^')

    plt.xlabel("Number of HMM Components")
    plt.ylabel("Distance / Divergence")
    plt.title("Effect of HMM Components on Distribution and Temporal Alignment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()