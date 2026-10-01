conda create -n fraud-detection-service-v1 python=3.12

conda activate fraud-detection-service-v1

uvicorn app.main:app --reload --port 8000

pytest tests/ -v