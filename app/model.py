import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from .utils import save_model, load_model, settings
MODEL_ISO = "iso.joblib"
MODEL_RF = "rf.joblib"
SCALER = "scaler.joblib"

def _prepare_matrix(df: pd.DataFrame, fit_scaler=False):
    X = df[['hour','path_depth','rpm','path_count','payload_len']].fillna(0)
    cats = pd.get_dummies(df[['status_cat','ua_cat']], drop_first=True)
    X = pd.concat([X.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    if fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        save_model(scaler, SCALER)
        return Xs
    else:
        scaler_path = os.path.join(settings.MODEL_DIR, SCALER)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler.transform(X)
        return X.values

def train_unsupervised(df: pd.DataFrame, n_estimators=200, contamination=0.01) -> Dict[str,Any]:
    X = _prepare_matrix(df, fit_scaler=True)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso.fit(X)
    save_model(iso, MODEL_ISO)
    return {"model":"isolationforest","samples":len(df)}

def score_unsupervised(df: pd.DataFrame) -> np.ndarray:
    iso_path = os.path.join(settings.MODEL_DIR, MODEL_ISO)
    if not os.path.exists(iso_path):
        raise FileNotFoundError("no iso model")
    iso = joblib.load(iso_path)
    X = _prepare_matrix(df, fit_scaler=False)
    raw = -iso.decision_function(X)
    arr = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return arr

def train_semi_supervised(df: pd.DataFrame, labels_df: pd.DataFrame, test_split=0.2) -> Dict[str,Any]:
    merged = df.merge(labels_df, on=['timestamp','source_ip'], how='left')
    merged['label'] = merged['label'].fillna(0).astype(int)
    X = _prepare_matrix(merged, fit_scaler=True)
    y = merged['label']
    if y.nunique() < 2:
        raise ValueError("Not enough target classes for semi-supervised training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) if len(X_test)>0 else None
    save_model(clf, MODEL_RF)
    return {"model":"randomforest","accuracy":acc,"samples":len(merged)}

def predict_semi(df: pd.DataFrame) -> np.ndarray:
    rf_path = os.path.join(settings.MODEL_DIR, MODEL_RF)
    if not os.path.exists(rf_path):
        raise FileNotFoundError("no rf model")
    clf = joblib.load(rf_path)
    X = _prepare_matrix(df, fit_scaler=False)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:,1]
    return clf.predict(X)

def explain_row(row: pd.Series) -> list:
    reasons = []
    if row.get('rpm',0) > 200:
        reasons.append("high_request_rate")
    if row.get('path_count',0) <= 2 and row.get('path_count',0) > 0:
        reasons.append("rare_path")
    if row.get('ua_cat') == 'bot':
        reasons.append("suspicious_user_agent")
    if row.get('path_depth',0) >= 5:
        reasons.append("deep_path")
    if row.get('status',0) >= 500:
        reasons.append("server_errors")
    return reasons

# Optional: if shap installed, produce per-row shap values (heavy)
def shap_explain(df: pd.DataFrame):
    try:
        import shap
    except Exception:
        raise RuntimeError("shap not installed")
    rf_path = os.path.join(settings.MODEL_DIR, MODEL_RF)
    if not os.path.exists(rf_path):
        raise FileNotFoundError("no rf model")
    clf = joblib.load(rf_path)
    X = _prepare_matrix(df, fit_scaler=False)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    return shap_vals
