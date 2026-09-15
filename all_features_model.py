import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/train_transaction.csv")

# Convert object columns to category for XGBoost
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    'max_depth':7,
    'learning_rate':0.05,
    'device': 'cuda',  
    'n_estimators':100,
    'enable_categorical': True
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
