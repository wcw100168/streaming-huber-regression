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
