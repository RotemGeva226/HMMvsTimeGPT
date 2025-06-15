import numpy as np
from hmmlearn.hmm import GaussianHMM

class Hmm:
    def __init__(self, n_components=4, n_iter=1000):
        self.model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=n_iter, init_params='',
                        params='stmc')

    def init_hmm_with_kmeans(self, means_init, covars_init):
        self.model.startprob_ = np.full(3, 1 / 3)
        self.model.transmat_ = np.full((3, 3), 1 / 3)
        self.model.means_ = means_init
        self.model.covars_ = covars_init

    def generate_sequences(self, start_price, n_steps, n_samples):
        """
          Generates multiple synthetic price sequences using a trained GaussianHMM.
          Parameters:
          - hmm_model: trained GaussianHMM model
          - start_price: float, starting price for all sequences (last real price before forecast)
          - n_steps: int, length of each forecast sequence
          - n_samples: int, number of sequences to generate

          Returns:
          - sequences: np.ndarray of shape (n_samples, n_steps + 1)
          - states: np.ndarray of shape (n_samples, n_steps)
          """
        sequences = []
        states = []
        for _ in range(n_samples):
            returns, state_seq = self.model.sample(n_steps)
            prices = np.insert(np.cumsum(returns), 0, start_price)
            sequences.append(prices)
            states.append(state_seq)
        return np.array(sequences), np.array(states)