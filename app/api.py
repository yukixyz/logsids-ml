import os
import io
import csv
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import pandas as pd
import aiofiles
from .utils import settings, LOG
from .preprocess import detect_and_parse, extract_features
from .model import train_unsupervised, score_unsupervised, train_semi_supervised, predict_semi, explain_row
from .dashboard import make_dashboard_html
from .rate_limiter import TokenBucket

app = FastAPI(title="log-ids-ml-advanced")

# simple per-IP tokenbuckets
buckets = {}
def get_bucket(ip):
    if ip not in buckets:
        buckets[ip] = TokenBucket(settings.RATE_MAX, settings.RATE_WINDOW)
    return buckets[ip]

def save_uploaded_file(path: str, contents: bytes):
    with open(path, 'wb') as f:
        f.write(contents)

@app.post("/ingest")
async def ingest(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    bucket = get_bucket(client_ip)
    if not bucket.consume():
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    dest = os.path.join(settings.UPLOAD_DIR, file.filename)
    # async write
    async with aiofiles.open(dest, 'wb') as f:
        await f.write(contents)
    # parse and store parquet
    try:
        df = detect_and_parse(dest)
        df = extract_features(df)
        parquet = dest + ".parquet"
        df.to_parquet(parquet)
    except Exception as e:
        LOG.exception("parse error")
        raise HTTPException(status_code=400, detail=f"parse error: {e}")
    return {"status":"ok","ingested":len(df),"parquet":os.path.basename(parquet)}

@app.post("/train")
async def train(background: BackgroundTasks, mode: str = Query("unsupervised", regex="^(unsupervised|semi)$")):
    # schedule training in background to avoid blocking
    pq_files = [f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(".parquet")]
    if not pq_files:
        raise HTTPException(status_code=400, detail="no data")
    latest = sorted(pq_files)[-1]
    df = pd.read_parquet(os.path.join(settings.UPLOAD_DIR, latest))
    if mode == "unsupervised":
        background.add_task(_train_unsupervised_task, df)
        return {"status":"training_started","mode":"unsupervised"}
    else:
        labels = os.path.join(settings.UPLOAD_DIR, "labels.csv")
        if not os.path.exists(labels):
            raise HTTPException(status_code=400, detail="labels.csv required for semi")
        labels_df = pd.read_csv(labels)
        background.add_task(_train_semi_task, df, labels_df)
        return {"status":"training_started","mode":"semi"}

def _train_unsupervised_task(df):
    LOG.info("Starting unsupervised training")
    res = train_unsupervised(df)
    LOG.info("Training finished: %s", res)

def _train_semi_task(df, labels_df):
    LOG.info("Starting semi-supervised training")
    try:
        res = train_semi_supervised(df, labels_df)
        LOG.info("Semi training finished: %s", res)
    except Exception as e:
        LOG.exception("Semi training failed: %s", e)

@app.get("/alerts")
def alerts(limit: int = 100, ip: str = None, use_semi: bool = False):
    pq_files = [f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(".parquet")]
    if not pq_files:
        return {"alerts":[]}
    latest = sorted(pq_files)[-1]
    df = pd.read_parquet(os.path.join(settings.UPLOAD_DIR, latest))
    try:
        df['anomaly_score'] = score_unsupervised(df)
    except FileNotFoundError:
        df['anomaly_score'] = 0.0
    if use_semi:
        try:
            df['semi_score'] = predict_semi(df)
        except FileNotFoundError:
            pass
    if ip:
        df = df[df['source_ip'] == ip]
    df = df.sort_values('anomaly_score', ascending=False).head(limit)
    out = []
    for _, row in df.iterrows():
        out.append({
            "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
            "source_ip": row['source_ip'],
            "path": row['path'],
            "status": int(row['status']),
            "anomaly_score": float(row['anomaly_score']),
            "reasons": explain_row(row)
        })
    return {"alerts": out}

@app.get("/report")
def report(format: str = Query("html", regex="^(html|pdf)$"), ip: str = None):
    pq_files = [f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(".parquet")]
    if not pq_files:
        raise HTTPException(status_code=400, detail="no data")
    latest = sorted(pq_files)[-1]
    df = pd.read_parquet(os.path.join(settings.UPLOAD_DIR, latest))
    try:
        df['anomaly_score'] = score_unsupervised(df)
    except FileNotFoundError:
        df['anomaly_score'] = 0.0
    html = make_dashboard_html(df, ip=ip)
    if format == "html":
        return HTMLResponse(content=html)
    else:
        pdf = HTML(string=html).write_pdf()
        return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf", headers={"Content-Disposition":"attachment; filename=report.pdf"})

@app.get("/health")
def health():
    models = []
    try:
        models = os.listdir(settings.MODEL_DIR)
    except Exception:
        models = []
    return {"status":"ok","models":models}
