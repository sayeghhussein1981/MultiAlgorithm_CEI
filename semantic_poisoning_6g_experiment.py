# ==============================================================================
# Title: Semantic Poisoning in 6G Networks - Experimental Code
# Description: This script reproduces the experimental results for semantic 
#              poisoning detection using the GenSC-6G dataset.
# Outputs: Table 1, Table 2, Z-test results, and ROC metrics.
# ==============================================================================

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------------------------
# 1. Data Loading and Feature Extraction
# ------------------------------------------------------------------------------
def load_and_extract_features():
    """
    Loads the GenSC-6G dataset and extracts semantic features.
    """
    print("Loading GenSC-6G dataset...")
    # Load the training split of the dataset
    dataset = load_dataset("CQILAB/GenSC-6G", split="train")
    
    features_clean, features_awgn10, features_awgn30, labels = [], [], [], []
    
    for sample in dataset:
        features_clean.append(np.array(sample["classification_efficientnet_features"], dtype=np.float32))
        features_awgn10.append(np.array(sample["classification_awgn10dB_efficientnet_features"], dtype=np.float32))
        features_awgn30.append(np.array(sample["classification_awgn30dB_efficientnet_features"], dtype=np.float32))
        labels.append(sample["classification_class"])
        
    return (np.array(features_clean), np.array(features_awgn10), 
            np.array(features_awgn30), np.array(labels))

# ------------------------------------------------------------------------------
# 2. Data Splitting
# ------------------------------------------------------------------------------
def split_data(features, labels, test_size=0.3):
    """
    Splits the data into training and testing sets.
    """
    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    return train_idx, test_idx

# ------------------------------------------------------------------------------
# 3. Semantic Poisoning Simulation
# ------------------------------------------------------------------------------
def inject_semantic_poisoning(features, labels, target_class=0, poison_ratio=0.3, shift_magnitude=2.0):
    """
    Injects structured, directional perturbations into the semantic feature space.
    """
    n_samples = len(features)
    n_poison = int(n_samples * poison_ratio)
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Calculate the direction vector (normalized difference between target class and global mean)
    target_mask = labels == target_class
    target_center = features[target_mask].mean(axis=0)
    global_mean = features.mean(axis=0)
    direction = (target_center - global_mean) / (np.linalg.norm(target_center - global_mean) + 1e-8)
    
    # Apply the perturbation
    poisoned_features = features.copy()
    std = features.std(axis=0).mean()
    for idx in poison_indices:
        poisoned_features[idx] = features[idx] + shift_magnitude * std * direction
        
    return poisoned_features

# ------------------------------------------------------------------------------
# 4. Model Training and Evaluation
# ------------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test_clean, X_test_attack, name="Attack"):
    """
    Trains a Random Forest IDS and evaluates it against a specific attack.
    Returns metrics and statistical test results.
    """
    # Build binary training set (0 = clean, 1 = small random perturbation)
    n_train = len(X_train)
    half = n_train // 2
    X_train_bin = np.vstack([X_train[:half], X_train[half:2*half] + np.random.normal(0, 0.1, X_train[half:2*half].shape)])
    y_train_bin = np.concatenate([np.zeros(half), np.ones(half)])
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_bin, y_train_bin)
    
    # Build test set
    n_test = len(X_test_clean)
    X_test_bin = np.vstack([X_test_clean, X_test_attack])
    y_test_bin = np.concatenate([np.zeros(n_test), np.ones(n_test)])
    
    # Predictions and Probabilities
    y_pred = clf.predict(X_test_bin)
    y_proba = clf.predict_proba(X_test_bin)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    auc = roc_auc_score(y_test_bin, y_proba)
    
    # Statistical Significance (One-sample Z-test against AUC = 0.5)
    # Using the standard error of AUC (Hanley & McNeil method approximation for binary classification)
    # Note: For a strict Z-test on AUC, we approximate the standard error.
    se_auc = np.sqrt((auc * (1 - auc)) / n_test) 
    z_score = (auc - 0.5) / se_auc
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) # Two-tailed test
    
    # 95% Confidence Interval for AUC (approximation)
    ci_lower = auc - 1.96 * se_auc
    ci_upper = auc + 1.96 * se_auc
    
    return {
        "Name": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC": auc,
        "p-value": p_value,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Load Data
    X_clean, X_awgn10, X_awgn30, y = load_and_extract_features()
    
    # 2. Split Data (70% Train, 30% Test)
    train_idx, test_idx = split_data(X_clean, y)
    
    X_train = X_clean[train_idx]
    X_test_clean = X_clean[test_idx]
    y_test = y[test_idx]
    
    X_test_awgn10 = X_awgn10[test_idx]
    X_test_awgn30 = X_awgn30[test_idx]
    
    # 3. Simulate Semantic Poisoning (Magnitude = 2.0 for baseline)
    X_test_poisoned = inject_semantic_poisoning(X_test_clean, y_test, shift_magnitude=2.0)
    
    # 4. Evaluate Table 1 (Baseline Results)
    print("\n--- Table 1: Detection Performance ---")
    results_t1 = []
    results_t1.append(train_and_evaluate(X_train, X_test_clean, X_test_poisoned, "Semantic Poisoning"))
    results_t1.append(train_and_evaluate(X_train, X_test_clean, X_test_awgn10, "AWGN 10 dB"))
    results_t1.append(train_and_evaluate(X_train, X_test_clean, X_test_awgn30, "AWGN 30 dB"))
    
    df_t1 = pd.DataFrame(results_t1)
    print(df_t1[["Name", "Accuracy", "F1-Score", "AUC"]].to_string(index=False))
    
    # 5. Evaluate Table 2 (Effect of Magnitude)
    print("\n--- Table 2: Effect of Poisoning Magnitude ---")
    magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    results_t2 = []
    
    for mag in magnitudes:
        X_test_poisoned_mag = inject_semantic_poisoning(X_test_clean, y_test, shift_magnitude=mag)
        res = train_and_evaluate(X_train, X_test_clean, X_test_poisoned_mag, f"Magnitude {mag}")
        results_t2.append(res)
        
    df_t2 = pd.DataFrame(results_t2)
    print(df_t2[["Name", "Accuracy", "F1-Score", "AUC"]].to_string(index=False))
    
    # 6. Print Statistical Significance (Table 3)
    print("\n--- Table 3: Statistical Significance (Z-test & 95% CI) ---")
    print(df_t1[["Name", "AUC", "p-value", "95% CI Lower", "95% CI Upper"]].to_string(index=False))
    print("\nNote: If the 95% CI includes 0.5 and p > 0.05, the result is statistically indistinguishable from random guessing.")
