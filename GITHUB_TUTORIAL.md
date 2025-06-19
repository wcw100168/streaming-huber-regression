# 串流 Huber 回歸套件 GitHub 發布與使用教學

## 目錄
1. [準備工作](#準備工作)
2. [上傳套件到 GitHub](#上傳套件到-github)
3. [從 GitHub 安裝使用套件](#從-github-安裝使用套件)
4. [套件使用教學](#套件使用教學)
5. [常見問題](#常見問題)

---

## 準備工作

### 1. 安裝必要工具

確保你的電腦已安裝：
- **Git**: [下載 Git](https://git-scm.com/download)
- **GitHub 帳號**: [註冊 GitHub](https://github.com)

### 2. 驗證安裝
```bash
# 檢查 Git 是否安裝成功
git --version

# 設定 Git 用戶資訊（如果還沒設定）
git config --global user.name "你的姓名"
git config --global user.email "你的信箱@example.com"
```

---

## 上傳套件到 GitHub

### 步驟 1: 創建 GitHub 儲存庫

1. 登入 [GitHub](https://github.com)
2. 點擊右上角的 "+" → "New repository"
3. 填寫儲存庫資訊：
   - **Repository name**: `streaming-huber-regression`
   - **Description**: `A Python package for streaming Huber regression with adaptive regularization`
   - **Public/Private**: 選擇 Public（讓其他人可以使用）
   - ✅ 勾選 "Add a README file"
   - ✅ 勾選 "Add .gitignore" → 選擇 "Python"
   - 可選擇 "Choose a license" → 建議選擇 "MIT License"
4. 點擊 "Create repository"

### 步驟 2: 初始化本地 Git 儲存庫

在終端機中執行：

```bash
# 進入套件目錄
cd "/Users/user/Downloads/巨量資料分析期末/實作嘗試v7/streaming_huber_regression"

# 初始化 Git 儲存庫
git init

# 添加遠端儲存庫（替換成你的 GitHub 用戶名）
git remote add origin https://github.com/你的用戶名/streaming-huber-regression.git
```

### 步驟 3: 創建 .gitignore 檔案

```bash
# 創建 .gitignore 檔案
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# macOS
.DS_Store

# Data files (可選，如果不想上傳大型資料檔案)
data/*.csv
*.csv
EOF
```

### 步驟 4: 添加和提交檔案

```bash
# 添加所有檔案到暫存區
git add .

# 提交檔案
git commit -m "Initial commit: Add streaming Huber regression package

- Core modules: trainer, standardizer, solvers
- Data loader with Dask support  
- Unit and integration tests
- Examples and documentation
- Complete package structure with setup.py"

# 推送到 GitHub
git push -u origin main
```

如果遇到認證問題，GitHub 現在需要使用 Personal Access Token：

1. 到 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 點擊 "Generate new token (classic)"
3. 勾選 "repo" 權限
4. 複製生成的 token
5. 在推送時使用 token 作為密碼

### 步驟 5: 創建發布版本（可選）

```bash
# 創建標籤
git tag -a v1.0.0 -m "Release version 1.0.0

Features:
- Streaming Huber regression with IRLS and LAMM solvers
- Online standardization with numerical stability
- Adaptive regularization with BIC model selection
- Memory-efficient data loading with Dask
- Comprehensive test suite
- Easy-to-use API with convenience functions"

# 推送標籤
git push origin v1.0.0
```

在 GitHub 網頁上：
1. 到你的儲存庫頁面
2. 點擊 "Releases" → "Create a new release"
3. 選擇剛創建的標籤 `v1.0.0`
4. 填寫發布說明
5. 點擊 "Publish release"

---

## 從 GitHub 安裝使用套件

### 方法 1: 直接從 GitHub 安裝（推薦）

```bash
# 使用 pip 直接從 GitHub 安裝
pip install git+https://github.com/你的用戶名/streaming-huber-regression.git

# 或安裝特定版本
pip install git+https://github.com/你的用戶名/streaming-huber-regression.git@v1.0.0
```

### 方法 2: Clone 後本地安裝

```bash
# Clone 儲存庫
git clone https://github.com/你的用戶名/streaming-huber-regression.git

# 進入目錄
cd streaming-huber-regression

# 安裝依賴
pip install -r requirements.txt

# 以開發模式安裝套件
pip install -e .
```

### 方法 3: 下載並安裝

```bash
# 下載 ZIP 檔案並解壓後
cd streaming-huber-regression-main

# 安裝
pip install .
```

---

## 套件使用教學

### 快速開始

```python
# 導入套件
from streaming_huber import streaming_huber_training

# 使用便利函數進行訓練
result = streaming_huber_training(
    data_file_path="your_data.csv",  # 你的資料檔案
    max_samples=10000,               # 訓練樣本數
    batch_size=1000,                 # 批次大小
    n_batch=10,                      # 批次數量
    tau_estimation_method='initial', # tau 估計方法
    penalty=False                    # 是否使用正則化
)

print(f"訓練完成，平均 MAE: {result['avg_mae']:.4f}")
```

### 詳細使用範例

```python
import numpy as np
from streaming_huber import (
    StreamingDataLoaderWithDask,
    StreamingHuberModelTrainer,
    OnlineStandardizer
)

# 1. 創建資料載入器
loader = StreamingDataLoaderWithDask(
    file_path="data.csv",
    batch_size=500,
    max_samples=5000
)

# 2. 創建訓練器
trainer = StreamingHuberModelTrainer(
    n_features=90,  # 根據你的資料特徵數調整
    tau_estimation_method='initial',
    penalty=False,
    auto_lambda=True
)

# 3. 批次訓練
mae_history = []
for i, batch in enumerate(loader):
    if batch is None:
        break
    
    X, y = batch
    result = trainer.train_on_batch(X, y)
    mae = result.get('mae', 0)
    mae_history.append(mae)
    
    print(f"批次 {i+1}: MAE = {mae:.4f}")

# 4. 進行預測
test_loader = StreamingDataLoaderWithDask("test_data.csv", batch_size=1000)
test_batch = test_loader.get_batch()
if test_batch:
    X_test, y_test = test_batch
    predictions = trainer.predict(X_test)
    test_mae = np.mean(np.abs(predictions - y_test))
    print(f"測試 MAE: {test_mae:.4f}")

# 5. 保存和載入模型
model_state = trainer.get_state()
# 可以保存到檔案: np.save('model_state.npy', model_state)

# 載入模型
new_trainer = StreamingHuberModelTrainer(n_features=90)
new_trainer.set_state(model_state)
```

### 進階使用：比較不同配置

```python
from streaming_huber import streaming_huber_training

# 測試不同配置
configs = [
    {"penalty": False, "name": "無正則化"},
    {"penalty": True, "auto_lambda": True, "name": "自動正則化"},
    {"penalty": True, "auto_lambda": False, "name": "固定正則化"}
]

results = {}
for config in configs:
    name = config.pop("name")
    result = streaming_huber_training(
        data_file_path="data.csv",
        max_samples=8000,
        batch_size=800,
        n_batch=10,
        **config
    )
    results[name] = result
    print(f"{name}: 平均 MAE = {result['avg_mae']:.4f}")

# 找出最佳配置
best_config = min(results.items(), key=lambda x: x[1]['avg_mae'])
print(f"最佳配置: {best_config[0]}")
```

---

## 常見問題

### Q1: 安裝時出現權限錯誤
```bash
# 使用 --user 參數
pip install --user git+https://github.com/你的用戶名/streaming-huber-regression.git

# 或使用虛擬環境
python -m venv streaming_huber_env
source streaming_huber_env/bin/activate  # Windows: streaming_huber_env\Scripts\activate
pip install git+https://github.com/你的用戶名/streaming-huber-regression.git
```

### Q2: NumPy 相容性警告
```bash
# 如果遇到 NumPy 2.x 相容性問題，可以降級
pip install "numpy<2.0"
pip install "pandas<2.0"
```

### Q3: 資料格式要求
- CSV 檔案，無標題行
- 第一列為目標變數（y）
- 其餘列為特徵變數（X）
- 數值資料，無缺失值

### Q4: 記憶體不足
```python
# 減少批次大小
loader = StreamingDataLoaderWithDask(
    file_path="data.csv",
    batch_size=100,  # 降低批次大小
    max_samples=1000
)
```

### Q5: 更新套件
```bash
# 更新到最新版本
pip install --upgrade git+https://github.com/你的用戶名/streaming-huber-regression.git
```

---

## 測試安裝

創建測試檔案 `test_installation.py`：

```python
#!/usr/bin/env python3
"""測試套件安裝是否成功"""

import numpy as np

def test_import():
    """測試模組導入"""
    try:
        from streaming_huber import (
            StreamingHuberModelTrainer,
            StreamingDataLoaderWithDask,
            OnlineStandardizer,
            streaming_huber_training
        )
        print("✅ 模組導入成功")
        return True
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        return False

def test_basic_functionality():
    """測試基本功能"""
    try:
        from streaming_huber import OnlineStandardizer
        
        # 測試標準化器
        standardizer = OnlineStandardizer(n_features=3)
        X = np.random.randn(100, 3)
        standardizer.update(X)
        X_std = standardizer.transform(X)
        
        print(f"✅ 基本功能測試通過")
        print(f"   原始資料形狀: {X.shape}")
        print(f"   標準化後形狀: {X_std.shape}")
        return True
    except Exception as e:
        print(f"❌ 基本功能測試失敗: {e}")
        return False

def test_synthetic_training():
    """測試合成資料訓練"""
    try:
        from streaming_huber import StreamingHuberModelTrainer
        
        # 生成合成資料
        np.random.seed(42)
        n_samples, n_features = 200, 5
        X = np.random.randn(n_samples, n_features)
        true_beta = np.random.randn(n_features)
        y = X @ true_beta + np.random.randn(n_samples) * 0.1
        
        # 訓練模型
        trainer = StreamingHuberModelTrainer(
            n_features=n_features,
            penalty=False
        )
        
        # 批次訓練
        batch_size = 50
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            trainer.train_on_batch(X_batch, y_batch)
        
        # 測試預測
        predictions = trainer.predict(X[:50])
        mae = np.mean(np.abs(predictions - y[:50]))
        
        print(f"✅ 合成資料訓練測試通過")
        print(f"   訓練樣本數: {n_samples}")
        print(f"   特徵維度: {n_features}")
        print(f"   測試 MAE: {mae:.4f}")
        return True
    except Exception as e:
        print(f"❌ 合成資料訓練測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("串流 Huber 回歸套件安裝測試")
    print("=" * 50)
    
    tests = [
        test_import,
        test_basic_functionality,
        test_synthetic_training
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"測試結果: {passed}/{len(tests)} 通過")
    if passed == len(tests):
        print("🎉 套件安裝成功，可以開始使用！")
    else:
        print("⚠️  部分測試失敗，請檢查安裝或依賴")
    print("=" * 50)
```

執行測試：
```bash
python test_installation.py
```

---

## 套件維護

### 更新套件
```bash
# 進入套件目錄
cd streaming-huber-regression

# 修改程式碼後
git add .
git commit -m "Update: 描述你的更改"
git push

# 創建新版本
git tag -a v1.0.1 -m "Bug fixes and improvements"
git push origin v1.0.1
```

### 分支管理
```bash
# 創建開發分支
git checkout -b develop

# 合併到主分支
git checkout main
git merge develop
git push
```

---

## 總結

這份教學涵蓋了：
1. ✅ **GitHub 上傳**: 從初始化到發布版本
2. ✅ **套件安裝**: 多種安裝方式
3. ✅ **使用教學**: 從基本到進階使用
4. ✅ **問題解決**: 常見問題和解決方案
5. ✅ **測試工具**: 驗證安裝是否成功

現在你可以將套件分享給其他人使用，或者在不同的環境中安裝使用你的串流 Huber 回歸套件！

**記得替換教學中的 `你的用戶名` 為你實際的 GitHub 用戶名。**
