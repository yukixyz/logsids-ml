import os, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .utils import save_model, load_model, MODEL_DIR
ISO_NAME = 'isolationforest.joblib'
RF_NAME = 'randomforest.joblib'
SCALER_NAME = 'scaler.joblib'
def _feature_matrix(df, fit_scaler=False):
    X = df[['hour','path_depth','rpm','path_count','payload_len']].fillna(0)
    cats = pd.get_dummies(df[['status_cat','ua_cat']], drop_first=True)
    X = pd.concat([X, cats.reset_index(drop=True)], axis=1)
    if fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        save_model(scaler, SCALER_NAME)
        return Xs
    else:
        scaler_path = os.path.join(MODEL_DIR, SCALER_NAME)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler.transform(X)
        else:
            return X.values
def train_unsupervised(df, n_estimators=200, contamination=0.01):
    X = _feature_matrix(df, fit_scaler=True)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso.fit(X)
    save_model(iso, ISO_NAME)
    return {'model': 'isolationforest', 'samples': len(df)}
def score_unsupervised(df):
    iso_path = os.path.join(MODEL_DIR, ISO_NAME)
    if not os.path.exists(iso_path):
        raise FileNotFoundError("IsolationForest model not found.")
    iso = joblib.load(iso_path)
    X = _feature_matrix(df, fit_scaler=False)
    raw = -iso.decision_function(X)
    arr = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return arr
def train_semi_supervised(df, labels_df, test_split=0.2):
    merged = df.merge(labels_df, on=['timestamp','source_ip'], how='left')
    merged['label'] = merged['label'].fillna(0).astype(int)
    X = _feature_matrix(merged, fit_scaler=True)
    y = merged['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y if y.nunique()>1 else None)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) if len(X_test)>0 else None
    save_model(clf, RF_NAME)
    return {'model':'randomforest', 'accuracy': score, 'samples': len(merged)}
def predict_semi(df):
    rf_path = os.path.join(MODEL_DIR, RF_NAME)
    if not os.path.exists(rf_path):
        raise FileNotFoundError("RandomForest model not found.")
    clf = joblib.load(rf_path)
    X = _feature_matrix(df, fit_scaler=False)
    probs = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X)
    return probs
def explain_reason(row):
    reasons = []
    if row.get('rpm',0) > 100:
        reasons.append('high_request_rate')
    if row.get('path_count',0) <= 2 and row.get('path_count',0) > 0:
        reasons.append('rare_path')
    if row.get('ua_cat') == 'bot':
        reasons.append('suspicious_user_agent')
    if 5 <= row.get('path_depth',0):
        reasons.append('deep_path_access')
    if row.get('status') >= 500:
        reasons.append('server_errors')
    return reasons
