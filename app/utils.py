import os, joblib, logging
from datetime import datetime
LOG = logging.getLogger("logids")
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR = os.path.join('/tmp', 'log_ids_uploads')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
def save_model(obj, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path)
    LOG.info("Saved model %s", path)
    return path
def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
