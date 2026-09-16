import pandas as pd

print("Loading data...")
df_trans = pd.read_csv('data/train_transaction.csv', nrows=10000) # Load subset for fast profiling
df_id = pd.read_csv('data/train_identity.csv', nrows=10000)

print("Transaction shape:", df_trans.shape)
print("Identity shape:", df_id.shape)

# How do they merge? Usually on TransactionID
if 'TransactionID' in df_trans.columns and 'TransactionID' in df_id.columns:
    df_merged = df_trans.merge(df_id, on='TransactionID', how='left')
    print("Merged shape:", df_merged.shape)
    
    # Check null percentages across merged
    null_pct = df_merged.isnull().mean()
    high_null_cols = null_pct[null_pct > 0.90].index.tolist()
    print(f"Number of columns with >90% nulls: {len(high_null_cols)}")
    print("Identity columns:", df_id.columns.tolist()[:10], "...")
    
else:
    print("No TransactionID found in one of the tables.")
