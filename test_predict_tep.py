import pandas as pd
import requests

csv_path = r"data/faulty_testing_fault1to20.csv"
df = pd.read_csv(csv_path)

# 取完整一個 run（前 960 筆）
records = df.head(960).to_dict(orient="records")

payload = {
    "records": records,
    "run_length": 960,
    "fault_start": 160,
    "aggregate": "max"
}

url = "http://localhost:8000/predict_tep_official_test_records"
resp = requests.post(url, json=payload)

print("Status code:", resp.status_code)

result = resp.json()
print("is_anomaly:", result.get("is_anomaly"))
print("num_windows:", result.get("num_windows"))
print("threshold:", result.get("threshold"))
print("max_window_score:", result.get("max_window_score"))
print("mean_window_score:", result.get("mean_window_score"))

# 看前後各 20 個 windows 的 label / prediction
print("first 20 window_labels:", result.get("window_labels", [])[:20])
print("first 20 window_predictions:", result.get("window_predictions", [])[:20])

print("last 20 window_labels:", result.get("window_labels", [])[-20:])
print("last 20 window_predictions:", result.get("window_predictions", [])[-20:])