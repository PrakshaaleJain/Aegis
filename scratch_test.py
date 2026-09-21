import pandas as pd
import numpy as np

# Mock data
df = pd.DataFrame({
    'UID': ['A', 'A', 'A', 'B', 'B'],
    'TransactionDT': [0, 2*86400, 10*86400, 0, 8*86400],
    'TransactionAmt': [100, 200, 50, 500, 10]
})

df_sorted = df.sort_values(['UID', 'TransactionDT'])
df_sorted.index = pd.to_timedelta(df_sorted['TransactionDT'], unit='s')

roll_7d = df_sorted.groupby('UID')['TransactionAmt'].rolling('7D').mean().reset_index(drop=True, level=0)
roll_30d = df_sorted.groupby('UID')['TransactionAmt'].rolling('30D').mean().reset_index(drop=True, level=0)

df_sorted['UID_TransactionAmt_mean_7d'] = roll_7d
df_sorted['UID_TransactionAmt_mean_30d'] = roll_30d

df_sorted['UID_TransactionAmt_EMA'] = df_sorted.groupby('UID')['TransactionAmt'].transform(lambda x: x.ewm(span=5).mean())

df_final = df_sorted.reset_index(drop=True)
print(df_final)
