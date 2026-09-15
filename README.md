# Aegis: Advanced Entity Resolution Fraud Detection

Aegis is a time-consistent machine learning pipeline for robust financial fraud detection. It moves beyond evaluating transactions in isolation by combining Entity Resolution, Client-Level Aggregations, and strict chronological Time-Gap validation to prevent temporal leakage and simulate real-world production environments. 

The pipeline is heavily inspired by the 1st and 2nd place solutions from the IEEE-CIS Fraud Detection Kaggle competition.

## 🚀 Core Strategies

### 1. Entity Resolution (Explicit UIDs)
Standard tabular datasets often anonymize user IDs. We reverse-engineer a unique human client representation (`UID`) by combining static user traits (like masked card IDs, address codes) with a relative exact registration day (`Dn`). This creates a highly accurate, unique proxy for a human client.

### 2. Client-Level Aggregations
Once we establish the `UID`, we force the tree models to understand the entity's behavior over time:
*   **Time-Deltas:** We calculate the exact time since this `UID`'s last transaction chronologically.
*   **Behavioral Stats:** We calculate the mean and standard deviation of continuous variables (like Transaction Amounts) grouped by `UID`.
*   **Frequency Encoding:** High-cardinality categorical variables are replaced with their raw dataset frequencies to neutralize train/test distribution shifts.

### 3. Time-Gap Validation Strategy
Standard 80/20 random splits ruin fraud models due to temporal leakage (the model effectively looks into the "future" to predict the "past"). 
Aegis splits the dataset chronologically:
*   **Train Fold:** e.g., Months 1-4.
*   **Gap (Skip):** e.g., Month 5 (Simulates the real-world delay in receiving confirmed chargeback fraud labels).
*   **Test Fold:** e.g., Month 6.

### 4. The 'Infected Card' Post-Processing Rule
After predicting the probabilities with our ensemble (LightGBM), we apply a killer business-logic rule: **If a `UID` hits a fraud probability > 0.90 at time $t$, that entity is permanently burned.** We programmatically force a 1.0 fraud probability on all future transactions for that specific `UID` in the test set.

## 📂 Repository Structure

*   `advanced_fraud_model.py`: The full implementation of the Aegis pipeline (UID generation, Aggregations, Time-Gap split, LightGBM training, and Post-Processing).
*   `all_features_model.py`: The original baseline XGBoost model.
*   `data/`: Directory for housing the transaction datasets (e.g., `train_transaction.csv`).
*   `requirements.txt`: Project dependencies.

## 🔮 Roadmap / Next Steps

*   **Differential Privacy:** Our immediate next step is to introduce differential privacy into the system to ensure that the reverse-engineered `UID`s and client-level behavioral aggregations do not leak sensitive PII while maintaining strong predictive performance. 