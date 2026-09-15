import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


def main():
    df = pd.read_csv("data/train_transaction.csv")
    df['TransactionDay'] = df['TransactionDT'] // (24 * 60 * 60)
    df['Dn'] = df['TransactionDay'] - df['D1']
    df['UID'] = df['card1'].astype(str) + '_' + \
                df['card2'].astype(str) + '_' + \
                df['card4'].astype(str) + '_' + \
                df['card6'].astype(str) + '_' + \
                df['addr1'].astype(str) + '_' + \
                df['Dn'].astype(str)
                
    print(f"Engineered UID column successfully. Unique UIDs found: {df['UID'].nunique()}")
    print("Sample of generated UIDs:")
    print(df[['card1', 'card2', 'card4', 'card6', 'addr1', 'Dn', 'UID']].head())

    print("Executing Step 2: Client-Level Aggregations...")
    
    # 2a. Group by UID: mean and std of TransactionAmt
    uid_amt_stats = df.groupby('UID')['TransactionAmt'].agg(['mean', 'std']).reset_index()
    uid_amt_stats.columns = ['UID', 'UID_TransactionAmt_mean', 'UID_TransactionAmt_std']
    df = df.merge(uid_amt_stats, on='UID', how='left')
    
    # 2b. Time-delta since UID's last transaction
    df = df.sort_values(['UID', 'TransactionDT'])
    df['UID_TimeSinceLastTrans'] = df.groupby('UID')['TransactionDT'].diff()
    df['UID_TimeSinceLastTrans'] = df['UID_TimeSinceLastTrans'].fillna(0)
    
    # Restore strictly chronological order for time-gap validation in Step 3
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    # 2c. Frequency Encoding for categoricals
    cat_cols = ['P_emaildomain', 'R_emaildomain', 'ProductCD', 'card4', 'card6']
    for col in cat_cols:
        freq_enc = df[col].value_counts(dropna=False).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_enc)
        
    print("Client-Level Aggregations completed.")
    print("Sample of new features:")
    print(df[['UID', 'TransactionAmt', 'UID_TransactionAmt_mean', 'UID_TransactionAmt_std', 'UID_TimeSinceLastTrans', 'P_emaildomain_freq']].head())

 
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])
    y = df['isFraud']

    # --- Step 3: Time-Gap Validation Strategy ---
    print("Executing Step 3: Time-Gap Validation Strategy...")
    # Dataset spans ~182 days. 
    # Train: Days 0-120 (Months 1-4)
    # Gap: Days 121-150 (Month 5) -> We skip these
    # Test: Days > 150 (Month 6)
    
    train_idx = df[df['TransactionDay'] <= 120].index
    test_idx = df[df['TransactionDay'] > 150].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


    print("Executing Step 4: Model Training (LightGBM)...")
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("Executing Post-Processing: Infected Card Rule...")
    test_df = df.loc[test_idx].copy()
    test_df['fraud_prob'] = y_pred_proba
    
    infected_uids = set()
    final_preds = []
    
    for idx, row in test_df.iterrows():
        uid = row['UID']
        prob = row['fraud_prob']
        
        # Don't infect missing/invalid UIDs
        if pd.isna(uid) or 'nan' in str(uid):
            final_preds.append(1 if prob >= 0.5 else 0)
            continue
            
        if uid in infected_uids:
            final_preds.append(1)
        else:
            if prob > 0.90:
                infected_uids.add(uid)
            final_preds.append(1 if prob >= 0.5 else 0)
            
    y_pred_post = final_preds

    print("--- Before Post-Processing ---")
    print(classification_report(y_test, y_pred))
    
    print("--- After Post-Processing ---")
    print(classification_report(y_test, y_pred_post))
    
    import numpy as np
    num_overrides = np.sum(np.array(y_pred_post) != np.array(y_pred))
    print(f"Number of 'Infected' overriding predictions: {num_overrides}")

if __name__ == "__main__":
    main()
