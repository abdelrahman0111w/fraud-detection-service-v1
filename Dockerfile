FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud_engine.py .
COPY app ./app
COPY model ./model

# Runtime state lives on a mounted volume so it survives restarts/redeploys.
RUN mkdir -p /srv/data
ENV FRAUD_DB_PATH=/srv/data/fraud_registry.db \
    FRAUD_AUDIT_LOG_PATH=/srv/data/audit_log.jsonl

EXPOSE 8000

# Railway (and most PaaS hosts) inject $PORT at runtime -- the container must
# listen on whatever port the platform assigns, not a hardcoded 8000. Falls
# back to 8000 for local `docker compose up`, where $PORT is unset.
# ONE worker on purpose: the SQLite velocity store is a local file, so extra
# workers/replicas would each see only part of the traffic. See README note.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]