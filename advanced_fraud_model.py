import pandas as pd
import numpy as np

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

    



if __name__ == "__main__":
    main()
