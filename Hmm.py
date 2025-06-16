import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

class Hmm:
    def __init__(self, n_components=3, n_iter=100, covariance_type='full', init_params='', params='stmc'):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            init_params=init_params,
            params=params,
            tol=1e-6,
            verbose=True
        )

    def init_startprob(self, startprob=None):
        if startprob is None:
            # Uniform distribution by default
            startprob = np.full(self.n_components, 1 / self.n_components)
        self.model.startprob_ = startprob

    def init_transmat(self, transmat=None):
        if transmat is None:
            # Uniform transition probabilities by default
            transmat = np.full((self.n_components, self.n_components), 1 / self.n_components)
        self.model.transmat_ = transmat

    def init_means(self, means_init):
        if means_init.shape[0] != self.n_components:
            raise ValueError("means_init must have shape (n_components, n_features)")
        self.model.means_ = means_init

    def init_covars(self, covars_init):
        if covars_init.shape[0] != self.n_components:
            raise ValueError("covars_init must have shape (n_components, n_features, n_features)")
        self.model.covars_ = covars_init

    def init_hmm_with_kmeans(self, means_init, covars_init, startprob=None, transmat=None):
        self.init_startprob(startprob)
        self.init_transmat(transmat)
        self.init_means(means_init)
        self.init_covars(covars_init)

    def generate_log_return_sequences(self, n_steps, n_sequences):
        sequences = []
        for _ in range(n_sequences):
            X, _ = self.model.sample(n_steps)  # X shape: (n_steps, 1), log returns
            log_returns = X.flatten()
            sequences.append(log_returns)
        return np.array(sequences)

    def plot_hmm_log_likelihood(self):
        if not hasattr(self.model, "monitor_"):
            raise ValueError("Model must be fitted before plotting log-likelihood.")

        log_likelihoods = self.model.monitor_.history
        iterations = list(range(len(log_likelihoods)))
        plt.figure()
        plt.plot(iterations, log_likelihoods, marker='o')
        plt.xticks(iterations)
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.title("HMM Training Log-Likelihood")
        plt.grid(True)
        plt.tight_layout()
        plt.show()