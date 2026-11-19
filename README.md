Semantic Poisoning in 6G Networks: A Threat Model for Semantic-Layer Attacks Against Intrusion Detection Systems

This repository contains the official reproducibility code and experimental setup for the paper: "Semantic Poisoning in 6G Networks: A Threat Model for Semantic-Layer Attacks Against Intrusion Detection Systems".

The code simulates semantic-layer poisoning attacks against a Random Forest-based Intrusion Detection System (IDS) using the GenSC-6G dataset. It evaluates the structural blindness of bit-level IDSs to semantic manipulations and compares the results against conventional Additive White Gaussian Noise (AWGN).

PREREQUISITES

To run the code, ensure you have Python 3.8+ installed. Install the required dependencies using pip:

pip install datasets scikit-learn pandas numpy statsmodels scipy

DATASET

The experiments utilize the GenSC-6G dataset, which is automatically downloaded via the Hugging Face datasets library upon running the code.

Dataset Repository: CQILAB/GenSC-6G (Hugging Face)

It provides 1,000-dimensional semantic feature representations for 6G semantic communication, including clean features and features corrupted by AWGN at 10 dB and 30 dB.

REPOSITORY CONTENTS

semantic_poisoning_6g_experiment.py: The main Python script that reproduces the experimental results.

README.md: This documentation file.

HOW TO RUN THE CODE

Clone this repository or download the source code file.

Open a terminal or command prompt.

Navigate to the directory containing the script.

Execute the script using Python:

python semantic_poisoning_6g_experiment.py

OUTPUTS AND REPRODUCIBILITY

When executed, the script performs the following steps and outputs the results directly to the console:

Data Loading & Splitting: Loads the GenSC-6G dataset and splits it into 70% training and 30% testing sets.

Semantic Poisoning Simulation: Injects structured, directional perturbations into the clean semantic features of the test set.

Model Training: Trains a Random Forest classifier (200 trees) to detect anomalies.

Table 1 Generation: Prints the detection performance (Accuracy, F1-Score, AUC) against Semantic Poisoning, AWGN 10 dB, and AWGN 30 dB.

Table 2 Generation: Prints the detection performance across different poisoning magnitudes (alpha = 0.5, 1.0, 2.0, 3.0, 5.0, 8.0). Note: alpha is written as a word to ensure compatibility with plain text editors.

Table 3 Generation: Performs one-sample Z-tests against the null hypothesis of random guessing (AUC = 0.5) and calculates the 95% confidence intervals for the baseline attacks.

NOTE ON RANDOM SEED

A fixed random seed (np.random.seed(42)) is set at the beginning of the script to guarantee the exact reproduction of the reported metrics in the manuscript.

