#!/bin/bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulate_attack.py --output sample_logs.csv --lines 5000
nohup uvicorn app.api:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
sleep 2
curl -F "file=@sample_logs.csv" http://127.0.0.1:8000/ingest
curl -X POST "http://127.0.0.1:8000/train?mode=unsupervised"
curl "http://127.0.0.1:8000/alerts?limit=10" > alerts.json
echo "alerts saved to alerts.json"
