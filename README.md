# Streaming Huber Regression

一個用於線上/串流 Huber 回歸的 Python 套件，支援自適應正則化和大規模資料處理。

## 特色功能

- 🚀 **串流處理**: 支援大規模資料的批次處理
- 📊 **線上標準化**: 自動更新統計量，無需預先載入全部資料
- 🎯 **自適應正則化**: 使用 LAMM 求解器和 BIC 準則自動選擇正則化參數
- 🔧 **彈性配置**: 支援多種 tau 估計方法和訓練模式
- 📈 **即時監控**: 內建訓練過程視覺化和統計分析

## 快速開始

### 安裝

```bash
pip install git+https://github.com/wcw100168/streaming-huber-regression.git
```
```bash
pip install -r requirements.txt
```

### 基本使用

```python
from streaming_huber import (
    StreamingDataLoaderWithDask,
    streaming_huber_training
)

# 建立資料載入器
loader = StreamingDataLoaderWithDask(
    file_path="data.csv",
    batch_size=1000,
    train_samples=100000
)

# 執行訓練
trainer, results = streaming_huber_training(
    data_loader=loader,
    n_features=90,
    penalty=True,
    auto_lambda=True
)

# 進行預測
X_test = ...  # 測試資料
y_pred = trainer.predict(X_test)
```

## API 文檔

### StreamingHuberModelTrainer

主要的串流 Huber 回歸訓練器。

#### 參數

- `n_features` (int): 特徵維度
- `tau_estimation_method` (str): tau 估計方法，可選 'initial', 'adaptive', 'fixed'
- `penalty` (bool): 是否使用 L1 正則化
- `auto_lambda` (bool): 是否自動選擇正則化參數

#### 方法

- `train_on_batch(X, y)`: 在單一批次上訓練模型
- `predict(X)`: 對新資料進行預測
- `reset()`: 重置模型狀態

### StreamingDataLoaderWithDask

高效能的串流資料載入器。

#### 參數

- `file_path` (str): 資料檔案路徑
- `batch_size` (int): 批次大小
- `train_samples` (int): 訓練樣本總數
- `buffer_batches` (int): 緩衝區批次數量

## 進階使用

### 使用不同配置

```python
from streaming_huber import StreamingHuberModelTrainer

# 不使用正則化
trainer = StreamingHuberModelTrainer(
    n_features=90, 
    penalty=False,
    tau_estimation_method='adaptive'
)

# 使用自動正則化
trainer = StreamingHuberModelTrainer(
    n_features=90, 
    penalty=True, 
    auto_lambda=True
)
```

### 自訂訓練循環

```python
trainer = StreamingHuberModelTrainer(n_features=90, penalty=True)

for epoch in range(10):
    while True:
        X_batch, y_batch = loader.get_batch()
        if X_batch is None:
            loader.reset()  # 重新開始
            break
        
        result = trainer.train_on_batch(X_batch, y_batch)
        print(f"Batch {result['batch_id']}: MAE = {result['mae']:.6f}")
```

## 測試

運行單元測試：

```bash
python tests/unit/test_standardizer.py
```

運行整合測試（需要 YearPredictionMSD.csv 資料）：

```bash
python tests/integration/test_yearprediction_msd.py
```

## 範例

查看 `examples/` 目錄中的完整範例：

- `basic_usage.py`: 基本使用方法和進階配置比較

## 套件結構

```
streaming_huber_regression/
├── streaming_huber/              # 主套件目錄
│   ├── __init__.py              # 套件初始化
│   ├── core/                    # 核心模組
│   │   ├── trainer.py           # StreamingHuberModelTrainer
│   │   └── standardizer.py     # OnlineStandardizer
│   ├── data/                    # 資料處理模組
│   │   └── loader.py            # StreamingDataLoaderWithDask
│   ├── solvers/                 # 求解器模組
│   │   ├── lamm.py              # LAMM 求解器
│   │   └── irls.py              # IRLS 求解器
│   └── utils/                   # 工具模組
│       └── training.py          # 訓練工具函數
├── tests/                       # 測試目錄
│   ├── unit/                    # 單元測試
│   └── integration/             # 整合測試
├── examples/                    # 範例目錄
├── setup.py                     # 套件安裝配置
├── requirements.txt             # 依賴套件
└── README.md                    # 專案說明
```

## 授權

MIT License
