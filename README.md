# log-ids-ml (Advanced)

Overview
--------
log-ids-ml is an advanced IDS prototype for ethical pentesting labs and ML experiments.

Quickstart
----------
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulate_attack.py --output sample_logs.csv --lines 5000
uvicorn app.api:app --reload --port 8000

API examples
------------
curl -F "file=@sample_logs.csv" http://127.0.0.1:8000/ingest
curl -X POST "http://127.0.0.1:8000/train?mode=unsupervised"
curl "http://127.0.0.1:8000/alerts?limit=10"
