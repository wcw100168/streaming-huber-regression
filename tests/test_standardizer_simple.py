"""
簡化的標準化器測試，避免複雜的依賴問題
"""

import sys
import os
import numpy as np

# 添加套件路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接導入標準化器，避免其他依賴
from streaming_huber.core.standardizer import OnlineStandardizer


def test_basic_functionality():
    """測試基本功能"""
    print("測試標準化器基本功能...")
    
    # 建立測試資料
    np.random.seed(42)
    X1 = np.random.randn(100, 5)
    X2 = np.random.randn(50, 5)
    
    # 建立標準化器
    standardizer = OnlineStandardizer(n_features=5)
    
    # 測試初始化
    assert standardizer.n_features == 5
    assert standardizer.count == 0
    print("✓ 初始化測試通過")
    
    # 測試更新
    standardizer.update(X1)
    assert standardizer.count == 100
    print("✓ 第一次更新測試通過")
    
    # 測試統計量
    mean, std = standardizer.get_stats()
    assert len(mean) == 5
    assert len(std) == 5
    assert np.all(std > 0)
    print("✓ 統計量計算測試通過")
    
    # 測試轉換
    X1_transformed = standardizer.transform(X1[:10])
    assert X1_transformed.shape == (10, 5)
    print("✓ 資料轉換測試通過")
    
    # 測試多批次更新
    standardizer.update(X2)
    assert standardizer.count == 150
    print("✓ 多批次更新測試通過")
    
    # 測試重置
    standardizer.reset()
    assert standardizer.count == 0
    assert np.allclose(standardizer.mean, 0)
    print("✓ 重置測試通過")
    
    print("🎉 所有基本功能測試通過！")


def test_numerical_accuracy():
    """測試數值精確性"""
    print("\n測試數值精確性...")
    
    # 建立已知統計量的資料
    np.random.seed(123)
    n_samples = 1000
    n_features = 3
    
    # 生成標準正態分佈資料
    X = np.random.randn(n_samples, n_features)
    
    # 使用線上標準化器
    standardizer = OnlineStandardizer(n_features)
    standardizer.update(X)
    
    mean_online, std_online = standardizer.get_stats()
    
    # 計算真實統計量
    mean_true = np.mean(X, axis=0)
    std_true = np.std(X, axis=0, ddof=1)
    
    # 檢查精確性
    mean_error = np.max(np.abs(mean_online - mean_true))
    std_error = np.max(np.abs(std_online - std_true))
    
    print(f"均值最大誤差: {mean_error:.2e}")
    print(f"標準差最大誤差: {std_error:.2e}")
    
    assert mean_error < 1e-10, f"均值誤差過大: {mean_error}"
    assert std_error < 1e-10, f"標準差誤差過大: {std_error}"
    
    print("✓ 數值精確性測試通過")


def test_batch_vs_incremental():
    """測試批次更新與逐一更新的一致性"""
    print("\n測試批次更新與逐一更新的一致性...")
    
    np.random.seed(456)
    X = np.random.randn(200, 4)
    
    # 批次更新
    standardizer_batch = OnlineStandardizer(4)
    standardizer_batch.update(X)
    mean_batch, std_batch = standardizer_batch.get_stats()
    
    # 逐一更新
    standardizer_incremental = OnlineStandardizer(4)
    for i in range(len(X)):
        standardizer_incremental.update(X[i:i+1])
    mean_incremental, std_incremental = standardizer_incremental.get_stats()
    
    # 檢查一致性
    mean_diff = np.max(np.abs(mean_batch - mean_incremental))
    std_diff = np.max(np.abs(std_batch - std_incremental))
    
    print(f"均值差異: {mean_diff:.2e}")
    print(f"標準差差異: {std_diff:.2e}")
    
    assert mean_diff < 1e-12, f"均值不一致: {mean_diff}"
    assert std_diff < 1e-12, f"標準差不一致: {std_diff}"
    
    print("✓ 批次與逐一更新一致性測試通過")


if __name__ == "__main__":
    print("="*60)
    print("串流 Huber 回歸套件 - 標準化器簡化測試")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_numerical_accuracy()
        test_batch_vs_incremental()
        
        print("\n" + "="*60)
        print("🎉 所有測試通過！標準化器功能正常")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
