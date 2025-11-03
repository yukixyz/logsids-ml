from app.preprocess import detect_and_parse, extract_features
import pandas as pd
def test_csv_and_features(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("timestamp,source_ip,method,path,status,user_agent\n2025-01-01 00:00:00,10.0.0.1,GET,/,200,Mozilla/5.0\n")
    df = detect_and_parse(str(p))
    df = extract_features(df)
    assert 'hour' in df.columns
    assert df.iloc[0]['is_private'] is True
