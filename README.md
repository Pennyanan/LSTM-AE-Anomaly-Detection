# LSTM-AE 異常偵測 API（Tennessee Eastman Process）

本專案提供一個基於 LSTM Autoencoder 的異常偵測服務，應用於 Tennessee Eastman Process（TEP）資料。  
系統已完成模型訓練與部署，使用者僅需提供資料，即可透過 API 完成前處理與異常偵測。

---

## 快速使用（Docker）

```bash
docker pull peianchen/tep-anomaly-api
docker run -p 8000:8000 peianchen/tep-anomaly-api
```
啟動後開啟：

http://localhost:8000/docs

Python 呼叫範例
```python
import pandas as pd
import requests

df = pd.read_csv("your_tep_test.csv")
records = df.head(960).to_dict(orient="records")

payload = {
    "records": records,
    "run_length": 960,
    "fault_start": 160,
    "aggregate": "max"
}

url = "http://localhost:8000/predict_tep_official_test_records"
resp = requests.post(url, json=payload)

print(resp.status_code)
print(resp.json())
```
專案特色
- 已訓練完成之 LSTM Autoencoder 模型
- 自動化前處理（符合 TEP 資料格式）
- Sliding Window 時序建模
- RESTful API（FastAPI）
- 支援 Docker 容器化部署
- 使用者無需重新訓練模型

### 系統架構
```
User Input (TEP Data)
        │
        ▼
FastAPI Service (app/main.py)
        │
        ▼
Preprocessing
   - 欄位篩選
   - Z-score 標準化
   - Sliding Window (L=20)
        │
        ▼
LSTM Autoencoder
        │
        ▼
Reconstruction Error
        │
        ▼
Threshold 判斷
        │
        ▼
Anomaly Result (API Response)
```
### 輸入資料格式

需為 TEP 格式資料，包含：
- xmeas_1 ~ xmeas_41
- xmv_1 ~ xmv_11

以下欄位若存在會自動忽略：

- faultNumber
- sample
- simulationRun
- label

### API 說明
GET /health

### 檢查服務狀態

POST /predict_tep_official_test_records

### 使用官方 TEP 測試資料格式進行預測

### 輸出內容
- is_anomaly：是否為異常
- aggregate_score：整體異常分數
- threshold：判斷門檻
- window_predictions：每個時間窗預測結果
- window_scores：每個時間窗異常分數
### 模型資訊
- 模型：LSTM Autoencoder
- 訓練方式：非監督式（僅使用正常資料）
- Window Size：20
- Threshold：由驗證集自動選取
### 系統流程
* 輸入原始 TEP 資料
* 自動進行欄位篩選
* Z-score 標準化（使用訓練統計量）
* Sliding Window 切分
* 計算 Reconstruction Error
* Threshold 判斷異常
### 專案結構
```
lstm-ae-service/
├── app/
├── model/
├── Dockerfile
├── requirements.txt
├── test_predict_tep.py
└── README.md
```
設計理念

本專案採用「訓練與推論分離」架構：

- 模型離線訓練
- API 僅負責推論
- 降低使用門檻，提升部署效率
