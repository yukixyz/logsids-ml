import os, io, csv, time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
import pandas as pd
from .preprocess import detect_and_parse, extract_features
from .model import train_unsupervised, score_unsupervised, train_semi_supervised, predict_semi, explain_reason
from .dashboard import make_dashboard_html
from .utils import UPLOAD_DIR, MODEL_DIR, save_model, load_model
from weasyprint import HTML
app = FastAPI(title="log-ids-ml")
RATE = {}
RATE_MAX = 10
RATE_WINDOW = 60
MAX_FILE_BYTES = 100 * 1024 * 1024
def check_rate(ip):
    now = time.time()
    arr = RATE.get(ip, [])
    arr = [t for t in arr if t > now - RATE_WINDOW]
    if len(arr) >= RATE_MAX:
        return False
    arr.append(now)
    RATE[ip] = arr
    return True
def last_parquet():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.parquet')]
    if not files:
        return None
    return os.path.join(UPLOAD_DIR, sorted(files)[-1])
@app.post('/ingest')
async def ingest(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    if not check_rate(client_ip):
        raise HTTPException(status_code=429, detail='rate limit exceeded')
    contents = await file.read()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='file too big')
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, 'wb') as f:
        f.write(contents)
    try:
        df = detect_and_parse(dest)
        df = extract_features(df)
        parquet = dest + '.parquet'
        df.to_parquet(parquet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'parse error: {e}')
    return {'status':'ok','ingested': len(df), 'parquet': os.path.basename(parquet)}
@app.post('/train')
def train(mode: str = Query('unsupervised', regex='^(unsupervised|semi)$')):
    pq = last_parquet()
    if not pq:
        raise HTTPException(status_code=400, detail='no data ingested')
    df = pd.read_parquet(pq)
    if mode == 'unsupervised':
    	res = train_unsupervised(df)
    	return {'status':'trained', 'result': res}
    else:
        labels_path = os.path.join(UPLOAD_DIR, 'labels.csv')
        if not os.path.exists(labels_path):
            raise HTTPException(status_code=400, detail='labels.csv required in upload dir for semi mode')
        labels_df = pd.read_csv(labels_path)
        if 'timestamp' in labels_df.columns:
            labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'], errors='coerce')
        res = train_semi_supervised(df, labels_df)
        return {'status':'trained', 'result': res}
@app.get('/alerts')
def get_alerts(min_score: float = 0.0, limit: int = 100, ip: Optional[str] = None, use_semi: bool = False):
    pq = last_parquet()
    if not pq:
        return {'alerts': []}
    df = pd.read_parquet(pq)
    try:
        scores = score_unsupervised(df)
        df['anomaly_score'] = scores
    except FileNotFoundError:
        df['anomaly_score'] = 0.0
    if use_semi:
        try:
            probs = predict_semi(df)
            df['semi_score'] = probs
        except FileNotFoundError:
            pass
    if ip:
        df = df[df['source_ip'] == ip]
    df = df.sort_values('anomaly_score', ascending=False).head(limit)
    alerts = []
    for _, row in df.iterrows():
        reasons = explain_reason(row)
        alerts.append({'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                       'source_ip': row['source_ip'],
                       'path': row['path'],
                       'status': int(row['status']),
                       'anomaly_score': float(row['anomaly_score']),
                       'reasons': reasons})
    return {'alerts': alerts}
@app.get('/alerts.csv')
def alerts_csv(min_score: float = 0.0, ip: Optional[str] = None):
    resp = get_alerts(min_score=min_score, limit=10000, ip=ip)
    alerts = resp.get('alerts', [])
    if not alerts:
        raise HTTPException(status_code=404, detail='no alerts')
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['timestamp','source_ip','path','status','anomaly_score','reasons'])
    for a in alerts:
        writer.writerow([a['timestamp'], a['source_ip'], a['path'], a['status'], a['anomaly_score'], "|".join(a['reasons'])])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=alerts.csv'})
@app.get('/report')
def report_pdf(ip: Optional[str] = None, format: str = Query('html', regex='^(html|pdf)$')):
    pq = last_parquet()
    if not pq:
        raise HTTPException(status_code=400, detail='no data')
    df = pd.read_parquet(pq)
    try:
        df['anomaly_score'] = score_unsupervised(df)
    except FileNotFoundError:
        df['anomaly_score'] = 0.0
    html = make_dashboard_html(df, ip=ip)
    if format == 'html':
        return HTMLResponse(content=html, status_code=200)
    else:
        pdf = HTML(string=html).write_pdf()
        return StreamingResponse(io.BytesIO(pdf), media_type='application/pdf', headers={'Content-Disposition':'attachment; filename=report.pdf'})
@app.get('/dashboard')
def dashboard(ip: Optional[str] = None):
    pq = last_parquet()
    if not pq:
        raise HTTPException(status_code=400, detail='no data')
    df = pd.read_parquet(pq)
    try:
        df['anomaly_score'] = score_unsupervised(df)
    except FileNotFoundError:
        df['anomaly_score'] = 0.0
    html = make_dashboard_html(df, ip=ip)
    return HTMLResponse(content=html)
@app.get('/health')
def health():
    return {'status':'ok', 'models': os.listdir(MODEL_DIR)}
