FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential libffi-dev libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
