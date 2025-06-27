# HMMvsTimeGPT
This repository, HMMvsTimeGPT, explores and compares the capabilities of Hidden Markov Models (HMMs) and TimeGPT models for time series analysis. The core objective is to investigate the process of reverse engineering an HMM and to evaluate the performance of a reconstructed model against the original, drawing parallels or distinctions with a modern time series forecasting model like TimeGPT.

## Project Description
The project focuses on a methodological approach to HMM reconstruction and comparison. It involves:
* Selection of an existing HMM as a ground truth.
* Sampling data sequences from this original model.
* Initializing and iteratively refining parameters of a new HMM using the Expectation-Maximization (EM) algorithm.
* Generating sequences from the newly trained HMM.
* Comparing the generated sequences from the original and trained HMM models, and potentially extending this comparison to sequences generated or handled the original model.

## Features
* Implementation of Hidden Markov Models `Hmm.py`.
* Integration or wrapper for TimeGPT functionality `TimeGPTWrapper.py`.
* Main script `main.py` orchestrating the model training, data generation, and comparison processes.
* Framework for comparing time series models using diverse metrics `utils.py`.

## Comparison Methodology
The comparison between the original and reconstructed/trained models, with TimeGPT, leverages a suite of quantitative metrics to assess the fidelity and performance:
* **Dynamic Time Warping (DTW):** To measure the similarity between two time series that may vary in speed or duration.
* **Wasserstein Distance (Earth Mover's Distance):** To quantify the dissimilarity between the probability distributions of the generated sequences.
* **Kullback-Leibler (KL) Divergence:** To measure the information gain when going from one probability distribution to another, assessing the information loss when approximating the original distribution with the learned model.




 