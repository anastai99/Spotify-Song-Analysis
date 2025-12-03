# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and do a minimal check. Adjust as needed."""
    df = pd.read_csv(path)
    # Basic checks
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df

def train_test_split_df(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame=None, numerical_cols=None):
    """Standard scale selected numeric columns. Returns scaler, X_train_scaled, X_val_scaled"""
    if numerical_cols is None:
        numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
    return scaler, X_train_scaled, X_val_scaled

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Saved model to {path}")

def load_model(path: str):
    return joblib.load(path)

def evaluate_classifier(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true))==2 else 'weighted')
    print("Accuracy:", acc)
    print("F1:", f1)
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:,1])
            print("ROC AUC:", auc)
        except Exception:
            pass
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
