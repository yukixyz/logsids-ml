import pandas as pd
import re
from datetime import datetime
from typing import Optional

COMMON_LOG_RE = re.compile(r'(?P<ip>\S+) \S+ \S+ \[(?P<time>.*?)\] "(?P<req>.*?)" (?P<status>\d{3}) (?P<size>\d+|-) "(?P<ref>.*?)" "(?P<ua>.*?)"')

def parse_common_log_line(line: str) -> Optional[dict]:
    m = COMMON_LOG_RE.match(line)
    if not m:
        return None
    ip = m.group('ip')
    time_str = m.group('time')
    try:
        timestamp = datetime.strptime(time_str.split(' ')[0], "%d/%b/%Y:%H:%M:%S")
    except Exception:
        timestamp = None
    req = m.group('req')
    try:
        method, path, _ = req.split(' ')
    except Exception:
        method, path = 'GET', '/'
    status = int(m.group('status'))
    ua = m.group('ua')
    return {'timestamp': timestamp, 'source_ip': ip, 'method': method, 'path': path, 'status': status, 'user_agent': ua}

def read_csv(path: str, gzip: bool=False) -> pd.DataFrame:
    df = pd.read_csv(path, compression='gzip' if gzip else None)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        raise ValueError("CSV must have 'timestamp' column")
    expected = ['source_ip','method','path','status','user_agent']
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"CSV missing column {c}")
    df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(0).astype(int)
    return df

def detect_and_parse(path: str) -> pd.DataFrame:
    try:
        return read_csv(path)
    except Exception:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = parse_common_log_line(line.strip())
                if parsed:
                    rows.append(parsed)
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        raise ValueError("Unsupported log format")

def is_private_ip(ip: str) -> bool:
    try:
        a,b = ip.split('.')[:2]
        a,b = int(a), int(b)
        if a == 10: return True
        if a == 192 and b == 168: return True
        if a == 172 and 16 <= b <= 31: return True
    except Exception:
        return False
    return False

def categorize_ua(ua: str) -> str:
    s = str(ua).lower()
    if any(x in s for x in ('bot','crawl','spider','sqlmap','nikto','nikto')):
        return 'bot'
    if any(x in s for x in ('mozilla','chrome','safari','edge')):
        return 'browser'
    return 'other'

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.floor('T')
    df['path_depth'] = df['path'].fillna('/').apply(lambda p: len(str(p).strip('/').split('/')) if p else 0)
    df['status_cat'] = df['status'].apply(lambda s: f"{s//100}xx")
    df['ua_cat'] = df['user_agent'].apply(categorize_ua)
    df['is_private'] = df['source_ip'].apply(is_private_ip)
    rpm = df.groupby(['source_ip','minute']).size().reset_index(name='rpm')
    df = df.merge(rpm, on=['source_ip','minute'], how='left')
    path_count = df['path'].value_counts().to_dict()
    df['path_count'] = df['path'].apply(lambda p: path_count.get(p,0))
    df['payload_len'] = df['path'].astype(str).apply(len) + df['user_agent'].astype(str).apply(len)
    return df
