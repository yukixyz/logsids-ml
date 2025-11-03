from app.model import train_unsupervised, score_unsupervised
from app.preprocess import extract_features
import pandas as pd
def test_model_basic(tmp_path):
    df = pd.DataFrame({
        'timestamp': [pd.to_datetime("2025-01-01 00:00:00")],
        'source_ip': ['10.0.0.1'],
        'method': ['GET'],
        'path': ['/'],
        'status': [200],
        'user_agent': ['Mozilla/5.0']
    })
    df = extract_features(df)
    res = train_unsupervised(df, n_estimators=10, contamination=0.5)
    scores = score_unsupervised(df)
    assert len(scores) == 1
