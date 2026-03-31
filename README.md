
```md
# LSTM-AE 異常偵測 API（Tennessee Eastman Process）

本專案提供一個基於 LSTM Autoencoder 的異常偵測服務，應用於 Tennessee Eastman Process（TEP）資料。  
系統已完成模型訓練與部署，使用者僅需提供資料，即可透過 API 完成前處理與異常偵測。

---

## 專案特色

- 已訓練完成之 LSTM Autoencoder 模型

- 自動化前處理（符合 TEP 資料格式）

- Sliding Window 時序建模

- RESTful API（FastAPI）

- 支援 Docker 容器化部署

- 使用者無需重新訓練模型


---

## 系統架構

User Input (TEP Data)
│
▼
FastAPI Service (app/main.py)
│
▼
Preprocessing

* 欄位篩選
* Z-score 標準化
* Sliding Window (L=20)
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

````

---

## 輸入資料格式

輸入需為 TEP 格式資料（CSV 或 JSON），包含以下欄位：

- xmeas_1 ~ xmeas_41
- xmv_1 ~ xmv_11

以下欄位若存在會自動忽略：

- faultNumber
- sample
- simulationRun
- label

---

## 本地執行方式

### 1. 安裝套件

```bash
pip install -r requirements.txt
````

### 2. 啟動 API

```bash
uvicorn app.main:app --reload
```

### 3. 開啟文件頁面

```
http://localhost:8000/docs
```

---

## Docker 執行方式

```bash
docker build -t lstm-ae-api .
docker run -p 8000:8000 lstm-ae-api
```

---

## API 說明

### GET /health

檢查服務是否正常運作

### POST /predict

進行異常偵測

---

## 使用範例

```python
import requests
import pandas as pd

df = pd.read_csv("tep_test.csv")

response = requests.post(
    "http://localhost:8000/predict",
    json=df.to_dict(orient="records")
)

print(response.json())
```

---

## 輸出內容

* is_anomaly：是否為異常
* aggregate_score：整體異常分數
* threshold：判斷門檻
* window_predictions：每個時間窗預測結果
* window_scores：每個時間窗異常分數

---

## 模型資訊

* 模型類型：LSTM Autoencoder
* 訓練方式：非監督式（僅使用正常資料）
* Window Size：20
* Threshold：由驗證集自動選取

---

## 系統流程

1. 輸入原始 TEP 資料
2. 自動進行欄位篩選
3. 使用訓練集統計量進行 Z-score 標準化
4. 建立 Sliding Window（時間序列切分）
5. 計算重建誤差（Reconstruction Error）
6. 根據閾值判斷是否為異常

---

## 專案結構

```
lstm-ae-service/
├── app/                # API 服務
├── model/              # 已訓練模型與相關檔案
├── Dockerfile
├── requirements.txt
├── test_predict_tep.py # 測試腳本
└── README.md
```

---

## 設計理念

本專案採用「訓練與推論分離」的設計：

* 模型訓練於離線完成
* API 僅負責推論與前處理
* 使用者無需具備機器學習背景即可使用
