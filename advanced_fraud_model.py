import pandas as pd
import numpy as np
import json
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.feature_selection import VarianceThreshold
import optuna

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    df = pd.read_csv(config['data']['train_path'])
    df['TransactionDay'] = df['TransactionDT'] // (24 * 60 * 60)
    df['Dn'] = df['TransactionDay'] - df['D1']
    for c in ['card1', 'card2', 'card4', 'card6', 'addr1', 'Dn']:
        df[c] = df[c].astype(str).replace('nan', 'NaN_val').fillna('NaN_val')
        
    df['UID'] = df['card1'] + '_' + df['card2'] + '_' + df['card4'] + '_' + df['card6'] + '_' + df['addr1'] + '_' + df['Dn']
                
    print(f"Engineered UID column successfully. Unique UIDs found: {df['UID'].nunique()}")
    print("Sample of generated UIDs:")
    print(df[['card1', 'card2', 'card4', 'card6', 'addr1', 'Dn', 'UID']].head())

    print("Executing Step 2: Client-Level Aggregations...")
    
    # 2a. Group by UID: mean and std of TransactionAmt
    # uid_amt_stats = df.groupby('UID')['TransactionAmt'].agg(['mean', 'std']).reset_index()
    # uid_amt_stats.columns = ['UID', 'UID_TransactionAmt_mean', 'UID_TransactionAmt_std']
    # df = df.merge(uid_amt_stats, on='UID', how='left')

    # We must sort by time and set a TimedeltaIndex for pandas rolling('7D') to work
    df = df.sort_values(['UID', 'TransactionDT'])
    df.index = pd.to_timedelta(df['TransactionDT'], unit='s')
    
    roll_7d = df.groupby('UID')['TransactionAmt'].rolling('7D').mean().reset_index(drop=True, level=0)
    roll_30d = df.groupby('UID')['TransactionAmt'].rolling('30D').mean().reset_index(drop=True, level=0)
    roll_7d_ewm = roll_7d.ewm(**config['ewm']).mean()
    roll_30d_ewm = roll_30d.ewm(**config['ewm']).mean()

    df['UID_TransactionAmt_roll_7d_ewm'] = roll_7d_ewm.values
    df['UID_TransactionAmt_roll_30d_ewm'] = roll_30d_ewm.values
    
    
    # 2b. Time-delta since UID's last transaction
    df['UID_TimeSinceLastTrans'] = df.groupby('UID')['TransactionDT'].diff()
    df['UID_TimeSinceLastTrans'] = df['UID_TimeSinceLastTrans'].fillna(0)
    
    # Restore strictly chronological order for time-gap validation in Step 3
    df = df.reset_index(drop=True).sort_values('TransactionDT').reset_index(drop=True)

    # 2c. Frequency Encoding for categoricals
    cat_cols = ['P_emaildomain', 'R_emaildomain', 'ProductCD', 'card4', 'card6']
    for col in cat_cols:
        freq_enc = df[col].value_counts(dropna=False).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_enc)
        
    print("Client-Level Aggregations completed.")
    print("Sample of new features:")
    print(df[['UID', 'TransactionAmt', 'UID_TransactionAmt_roll_7d_ewm', 'UID_TransactionAmt_roll_30d_ewm', 'UID_TimeSinceLastTrans', 'P_emaildomain_freq']].head())

 
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
    
    train_idx = df[df['TransactionDay'] <= config['validation']['train_max_day']].index
    test_idx = df[df['TransactionDay'] > config['validation']['test_min_day']].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Feature Selection Pipeline ---
    print("Executing Feature Selection Pipeline...")
    v_cols = [c for c in X_train.columns if c.startswith('V')]
    print(f"Initial V-columns: {len(v_cols)}")
    
    fs_config = config.get('feature_selection', {})
    null_thresh = fs_config.get('null_threshold', 0.5)
    var_thresh = fs_config.get('variance_threshold', 0.01)
    corr_thresh = fs_config.get('corr_threshold', 0.95)
    top_k = fs_config.get('top_k_features', 100)
    
    # Stage A: Drop high nullity V-columns
    null_pct = X_train[v_cols].isnull().mean()
    cols_to_drop_null = null_pct[null_pct > null_thresh].index.tolist()
    X_train = X_train.drop(columns=cols_to_drop_null)
    X_test = X_test.drop(columns=cols_to_drop_null)
    v_cols = [c for c in v_cols if c not in cols_to_drop_null]
    print(f"Stage A: Dropped {len(cols_to_drop_null)} V-columns with >{null_thresh*100}% nulls. Remaining: {len(v_cols)}")

    # Stage B: Drop near-zero variance V-columns
    vt = VarianceThreshold(threshold=var_thresh)
    vt.fit(X_train[v_cols].fillna(-999))
    cols_to_keep_var = np.array(v_cols)[vt.get_support()]
    cols_to_drop_var = set(v_cols) - set(cols_to_keep_var)
    X_train = X_train.drop(columns=list(cols_to_drop_var))
    X_test = X_test.drop(columns=list(cols_to_drop_var))
    v_cols = list(cols_to_keep_var)
    print(f"Stage B: Dropped {len(cols_to_drop_var)} V-columns with <{var_thresh} variance. Remaining: {len(v_cols)}")

    # Stage C: Correlation-based deduplication
    corr_matrix = X_train[v_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    X_train = X_train.drop(columns=cols_to_drop_corr)
    X_test = X_test.drop(columns=cols_to_drop_corr)
    v_cols = [c for c in v_cols if c not in cols_to_drop_corr]
    print(f"Stage C: Dropped {len(cols_to_drop_corr)} V-columns with >{corr_thresh} correlation. Remaining: {len(v_cols)}")
    
    # Stage D: LightGBM Importance Selection
    print("Stage D: Training quick LightGBM for feature importance...")
    temp_model = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    temp_model.fit(X_train, y_train)
    importances = temp_model.feature_importances_
    
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    selected_features = feat_imp.head(top_k).index.tolist()
    dropped_by_importance = set(X_train.columns) - set(selected_features)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"Stage D: Kept top {len(selected_features)} features globally based on importance. Dropped {len(dropped_by_importance)} features.")
    print(f"Final feature count for training: {X_train.shape[1]}")

    print("Executing Step 4: Optuna Hyperparameter Tuning...")
    optuna_train_mask = df.loc[train_idx, 'TransactionDay'] <= config['optuna']['train_max_day']
    optuna_val_mask = (df.loc[train_idx, 'TransactionDay'] >= config['optuna']['val_min_day']) & (df.loc[train_idx, 'TransactionDay'] <= config['optuna']['val_max_day'])
    
    X_opt_train, y_opt_train = X_train[optuna_train_mask], y_train[optuna_train_mask]
    X_opt_val, y_opt_val = X_train[optuna_val_mask], y_train[optuna_val_mask]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model_opt = LGBMClassifier(**params)
        model_opt.fit(X_opt_train, y_opt_train)
        preds_proba = model_opt.predict_proba(X_opt_val)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(y_opt_val, preds_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        return np.max(f1_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config['optuna']['n_trials'], timeout=config['optuna']['timeout'])
    
    print(f"Best Trial F1: {study.best_value:.4f}")
    print("Best Params:", study.best_params)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    print("Executing Step 5: Final Model Training (LightGBM)...")
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Finding optimal threshold using Precision-Recall curve on training data...")
    y_train_proba = model.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else config['post_processing']['prediction_threshold']
    print(f"Optimal threshold found (max F1 on train): {optimal_threshold:.4f}")
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    print("Executing Post-Processing: Infected Card Rule...")
    test_df = df.loc[test_idx].copy()
    test_df['fraud_prob'] = y_pred_proba
    
    infected_uids = {}
    final_preds = []
    
    for idx, row in test_df.iterrows():
        uid = row['UID']
        prob = row['fraud_prob']
        txn_day = row['TransactionDay']
        expiry_day = config['post_processing']['infected_card_expiry_days']
        if pd.isna(uid) or 'nan' in str(uid):
            final_preds.append(1 if prob >= optimal_threshold else 0)
            continue

        if uid in infected_uids:
            if txn_day - infected_uids[uid] <= expiry_day:
                final_preds.append(1)
                infected_uids[uid] = txn_day
                continue
            else:
                del infected_uids[uid]
                
        if prob > config['post_processing']['infected_card_threshold']:
            infected_uids[uid] = txn_day
        final_preds.append(1 if prob >= optimal_threshold else 0)
            
    y_pred_post = final_preds

    print("--- Before Post-Processing ---")
    print(classification_report(y_test, y_pred))
    
    print("--- After Post-Processing ---")
    print(classification_report(y_test, y_pred_post))
    num_overrides = np.sum(np.array(y_pred_post) != np.array(y_pred))
    print(f"Number of 'Infected' overriding predictions: {num_overrides}")

if __name__ == "__main__":
    main()
