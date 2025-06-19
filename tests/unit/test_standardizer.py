"""
線上標準化器單元測試
"""

import pytest
import numpy as np
import sys
import os

# 添加套件路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from streaming_huber.core.standardizer import OnlineStandardizer


class TestOnlineStandardizer:
    """線上標準化器測試"""
    
    def test_initialization(self):
        """測試初始化"""
        standardizer = OnlineStandardizer(n_features=5)
        assert standardizer.n_features == 5
        assert standardizer.count == 0
        assert len(standardizer.mean) == 5
        assert len(standardizer.M2) == 5
        
    def test_reset(self):
        """測試重置功能"""
        standardizer = OnlineStandardizer(n_features=3)
        
        # 先更新一些資料
        X = np.random.randn(10, 3)
        standardizer.update(X)
        assert standardizer.count == 10
        
        # 重置
        standardizer.reset()
        assert standardizer.count == 0
        assert np.allclose(standardizer.mean, 0)
        assert np.allclose(standardizer.M2, 0)
        
    def test_update_single_batch(self):
        """測試單批次更新"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        standardizer = OnlineStandardizer(n_features=5)
        
        standardizer.update(X)
        assert standardizer.count == 100
        
        mean, std = standardizer.get_stats()
        assert len(mean) == 5
        assert len(std) == 5
        
        # 檢查均值和標準差的合理性
        assert np.all(std > 0)  # 標準差應該大於0
        
    def test_update_multiple_batches(self):
        """測試多批次更新"""
        np.random.seed(42)
        X1 = np.random.randn(50, 3)
        X2 = np.random.randn(30, 3)
        
        standardizer = OnlineStandardizer(n_features=3)
        
        # 分批更新
        standardizer.update(X1)
        standardizer.update(X2)
        
        assert standardizer.count == 80
        
        # 與一次性更新比較
        X_all = np.vstack([X1, X2])
        standardizer_ref = OnlineStandardizer(n_features=3)
        standardizer_ref.update(X_all)
        
        mean1, std1 = standardizer.get_stats()
        mean2, std2 = standardizer_ref.get_stats()
        
        # 結果應該相同
        np.testing.assert_array_almost_equal(mean1, mean2, decimal=10)
        np.testing.assert_array_almost_equal(std1, std2, decimal=10)
        
    def test_transform(self):
        """測試資料轉換"""
        np.random.seed(42)
        X_train = np.random.randn(200, 4) * 2 + 5  # 均值=5, 標準差約=2
        X_test = np.random.randn(50, 4) * 2 + 5
        
        standardizer = OnlineStandardizer(n_features=4)
        standardizer.update(X_train)
        
        # 轉換測試資料
        X_test_transformed = standardizer.transform(X_test)
        
        assert X_test_transformed.shape == X_test.shape
        
        # 檢查轉換後的統計特性（應該接近標準化）
        # 注意：由於 X_test 是獨立生成的，不會完全標準化
        
    def test_get_stats_small_sample(self):
        """測試小樣本情況下的統計量"""
        standardizer = OnlineStandardizer(n_features=2)
        
        # 沒有資料時
        mean, std = standardizer.get_stats()
        assert np.allclose(mean, 0)
        assert np.allclose(std, 1)
        
        # 只有一個樣本時
        X = np.array([[1, 2]])
        standardizer.update(X)
        mean, std = standardizer.get_stats()
        assert np.allclose(mean, [1, 2])
        assert np.allclose(std, 1)  # 應該返回1避免除零
        
    def test_numerical_stability(self):
        """測試數值穩定性"""
        # 測試極大值
        X_large = np.array([[1e10, 1e10], [1e10 + 1, 1e10 + 1]])
        standardizer = OnlineStandardizer(n_features=2)
        standardizer.update(X_large)
        
        mean, std = standardizer.get_stats()
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std > 0)
        
        # 測試極小值
        X_small = np.array([[1e-10, 1e-10], [2e-10, 2e-10]])
        standardizer.reset()
        standardizer.update(X_small)
        
        mean, std = standardizer.get_stats()
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std > 0)


if __name__ == "__main__":
    # 運行測試
    test_class = TestOnlineStandardizer()
    
    print("運行線上標準化器測試...")
    
    try:
        test_class.test_initialization()
        print("✓ 初始化測試通過")
        
        test_class.test_reset()
        print("✓ 重置測試通過")
        
        test_class.test_update_single_batch()
        print("✓ 單批次更新測試通過")
        
        test_class.test_update_multiple_batches()
        print("✓ 多批次更新測試通過")
        
        test_class.test_transform()
        print("✓ 資料轉換測試通過")
        
        test_class.test_get_stats_small_sample()
        print("✓ 小樣本統計量測試通過")
        
        test_class.test_numerical_stability()
        print("✓ 數值穩定性測試通過")
        
        print("\n所有測試通過！")
        
    except Exception as e:
        print(f"✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
