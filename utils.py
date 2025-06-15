import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

def plot_data(generated_sequences: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for source, group in generated_sequences.groupby("source"):
        plt.plot(group["ds"], group["y"], label=source)
    plt.legend()
    plt.title("Historical vs Forecasted Stock Prices (AAPL)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def initialize_with_kmeans(forecast_sequences: np.ndarray, n_states: int = 4):
    """
    This function returns:
    X - vertical stack of log returns (price differences) from all the forecasted sequences.
    Each row in X represents one return value (daily change in stock price).
    means_init - The mean return of each hidden state.
    covars_init - 3D array with the variance for each hidden state.
    """
    # Step 1: Convert each forecast sequence to returns
    return_seqs = [np.diff(seq).reshape(-1, 1) for seq in forecast_sequences]
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