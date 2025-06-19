"""
簡化的整合測試 - 測試核心功能整合
"""

import os
import pytest
import numpy as np
import tempfile

# 設定路徑以便導入模組
import sys
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
streaming_huber_dir = os.path.join(project_root, "streaming_huber_regression")
if streaming_huber_dir not in sys.path:
    sys.path.insert(0, streaming_huber_dir)

from streaming_huber.data.loader import StreamingDataLoaderWithDask
from streaming_huber.core.trainer import StreamingHuberModelTrainer
from streaming_huber.utils.training import streaming_huber_training


class TestStreamingHuberIntegration:
    """串流 Huber 回歸整合測試"""
    
    @classmethod
    def setup_class(cls):
        """設置測試環境"""
        cls.data_file_path = os.path.join(os.path.dirname(project_root), 'data/YearPredictionMSD.csv')
        cls.n_samples = 1000  # 小樣本測試
        cls.batch_size = 100
        cls.n_features = 90
    
    def test_data_file_exists(self):
        """測試資料檔案是否存在"""
        assert os.path.exists(self.data_file_path), f"資料檔案不存在: {self.data_file_path}"
    
    def test_data_loader_basic(self):
        """測試資料載入器基本功能"""
        if not os.path.exists(self.data_file_path):
            pytest.skip("資料檔案不存在，跳過測試")
        
        loader = StreamingDataLoaderWithDask(
            self.data_file_path,
            batch_size=self.batch_size,
            max_samples=self.n_samples
        )
        
        # 測試獲取一個批次
        batch = loader.get_batch()
        assert batch is not None
        assert len(batch) == 2  # X, y
        
        X, y = batch
        assert X.shape[1] == self.n_features
        assert len(y) == len(X)
    
    def test_trainer_basic(self):
        """測試訓練器基本功能"""
        if not os.path.exists(self.data_file_path):
            pytest.skip("資料檔案不存在，跳過測試")
        
        # 創建訓練器
        trainer = StreamingHuberModelTrainer(
            n_features=self.n_features,
            tau_estimation_method='initial',
            penalty=False,  # 簡化測試，不使用正則化
            auto_lambda=False
        )
        
        # 載入少量資料進行測試
        loader = StreamingDataLoaderWithDask(
            self.data_file_path,
            batch_size=self.batch_size,
            max_samples=self.n_samples
        )
        
        # 訓練幾個批次
        batch_count = 0
        for batch in loader:
            if batch is None:
                break
            
            X, y = batch
            trainer.train_on_batch(X, y)
            batch_count += 1
            
            if batch_count >= 3:  # 只測試 3 個批次
                break
        
        assert batch_count > 0, "沒有處理任何批次"
        assert trainer.batch_count > 0, "訓練器沒有看到任何樣本"
        
        # 測試預測
        X, y = loader.get_batch()
        if X is not None:
            predictions = trainer.predict(X[:10])  # 預測前 10 個樣本
            assert len(predictions) == 10
            assert not np.any(np.isnan(predictions)), "預測結果包含 NaN"
    
    def test_convenience_function(self):
        """測試便利函數"""
        if not os.path.exists(self.data_file_path):
            pytest.skip("資料檔案不存在，跳過測試")
        
        # 使用便利函數進行快速訓練
        result = streaming_huber_training(
            data_file_path=self.data_file_path,
            max_samples=self.n_samples,
            batch_size=self.batch_size,
            n_batch=3,  # 只訓練 3 個批次
            tau_estimation_method='initial',
            penalty=False
        )
        
        assert result is not None
        assert 'trainer' in result
        assert 'train_rmse' in result
        assert result['trainer'].batch_count > 0
    
    def test_model_persistence(self):
        """測試模型持久化"""
        if not os.path.exists(self.data_file_path):
            pytest.skip("資料檔案不存在，跳過測試")
        
        # 訓練一個簡單模型
        trainer = StreamingHuberModelTrainer(
            n_features=self.n_features,
            tau_estimation_method='initial',
            penalty=False
        )
        
        loader = StreamingDataLoaderWithDask(
            self.data_file_path,
            batch_size=self.batch_size,
            max_samples=200
        )
        
        X, y = loader.get_batch()
        if X is not None:
            trainer.train_on_batch(X, y)
        
        # 測試狀態保存和載入
        state = trainer.get_state()
        assert state is not None
        assert 'beta' in state
        
        # 創建新訓練器並載入狀態
        new_trainer = StreamingHuberModelTrainer(
            n_features=self.n_features,
            tau_estimation_method='initial',
            penalty=False
        )
        new_trainer.set_state(state)
        
        # 驗證狀態一致性
        assert np.allclose(trainer.beta, new_trainer.beta)
        assert trainer.batch_count == new_trainer.batch_count


def test_synthetic_data():
    """使用合成資料的快速測試"""
    np.random.seed(42)
    n_samples, n_features = 500, 10
    
    # 生成合成資料
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features)
    noise = np.random.randn(n_samples) * 0.1
    y = X @ true_beta + noise
    
    # 創建訓練器（不使用正則化以避免 LAMM 複雜性）
    trainer = StreamingHuberModelTrainer(
        n_features=n_features,
        tau_estimation_method='initial',
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
    predictions = trainer.predict(X[:100])
    rmse = np.sqrt(np.mean((predictions - y[:100]) ** 2))
    
    assert len(predictions) == 100
    assert rmse < 2.0  # 合理的 RMSE 閾值
    assert trainer.batch_count == n_samples // batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
