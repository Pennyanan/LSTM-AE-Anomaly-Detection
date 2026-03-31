# TEP Anomaly Detection API

## Run API
docker build -t tep .
docker run -p 8000:8000 tep

## Predict
python test_predict_tep.py