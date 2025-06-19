"""
使用 YearPredictionMSD 資料集的完整整合測試

這個測試使用真實的 YearPredictionMSD 資料集來驗證串流 Huber 回歸模型的完整功能。
測試內容包括：
1. 資料載入和預處理
2. 串流訓練過程
3. 模型預測和評估
4. 正則化和非正則化版本的比較
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, Dict, Any, List

# 添加套件路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from streaming_huber.data.loader import StreamingDataLoaderWithDask
from streaming_huber.core.trainer import StreamingHuberModelTrainer
from streaming_huber.utils.training import streaming_huber_training


class TestYearPredictionMSDIntegration:
    """YearPredictionMSD 資料集整合測試"""
    
    @classmethod
    def setup_class(cls):
        """類別初始化設置"""
        cls.data_file_path = os.path.join(os.path.dirname(__file__), '../../data/YearPredictionMSD.csv')
        cls.n_features = 90  # YearPredictionMSD 有 90 個特徵
        
        # 測試參數
        cls.train_samples = 50000  # 減少樣本數以加快測試
        cls.batch_size = 1000
        cls.n_batch = 50
        
        print(f"初始化 YearPredictionMSD 整合測試")
        print(f"資料檔案: {cls.data_file_path}")
        print(f"訓練樣本數: {cls.train_samples:,}")
        print(f"批次大小: {cls.batch_size}")
        print(f"批次數量: {cls.n_batch}")
    
    def verify_data_file(self) -> bool:
        """驗證資料檔案是否存在且格式正確"""
        if not os.path.exists(self.data_file_path):
            print(f"錯誤: 資料檔案不存在: {self.data_file_path}")
            return False
        
        try:
            # 嘗試載入一小部分資料來驗證格式
            import pandas as pd
            sample_data = pd.read_csv(self.data_file_path, header=None, nrows=10)
            
            if sample_data.shape[1] != self.n_features + 1:  # +1 for target
                print(f"錯誤: 資料維度不正確。期望 {self.n_features + 1} 列，實際 {sample_data.shape[1]} 列")
                return False
            
            print(f"✓ 資料檔案驗證通過，形狀: {sample_data.shape}")
            return True
            
        except Exception as e:
            print(f"錯誤: 無法載入資料檔案: {e}")
            return False
    
    def test_data_loader(self) -> bool:
        """測試資料載入器功能"""
        print("\n=== 測試資料載入器 ===")
        
        try:
            loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            # 測試獲取批次
            batch_count = 0
            total_samples = 0
            
            while batch_count < 5:  # 只測試前 5 個批次
                X_batch, y_batch = loader.get_batch()
                if X_batch is None:
                    break
                
                batch_count += 1
                total_samples += len(X_batch)
                
                # 驗證批次形狀
                assert X_batch.shape[1] == self.n_features, f"特徵維度不正確: {X_batch.shape[1]}"
                assert len(y_batch) == len(X_batch), "特徵和標籤數量不匹配"
                
                print(f"批次 {batch_count}: {X_batch.shape}, 標籤範圍: [{y_batch.min():.1f}, {y_batch.max():.1f}]")
            
            # 測試統計信息
            stats = loader.get_loader_stats()
            print(f"載入器統計: {stats}")
            
            # 測試測試資料載入
            X_test, y_test = loader.get_test_data()
            if X_test is not None:
                print(f"測試資料載入成功: {X_test.shape}")
            
            loader.close()
            print("✓ 資料載入器測試通過")
            return True
            
        except Exception as e:
            print(f"✗ 資料載入器測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_streaming_training_no_penalty(self) -> Tuple[bool, Dict[str, Any]]:
        """測試不帶正則化的串流訓練"""
        print("\n=== 測試串流訓練（無正則化）===")
        
        try:
            # 建立資料載入器
            loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            start_time = time.time()
            
            # 執行訓練
            trainer, results = streaming_huber_training(
                data_loader=loader,
                n_features=self.n_features,
                n_batch=self.n_batch,
                penalty=False,
                tau_estimation='adaptive',
                mode='fast',
                verbose=True
            )
            
            training_time = time.time() - start_time
            
            # 驗證結果
            assert len(results) == self.n_batch, f"結果數量不正確: {len(results)}"
            assert trainer.is_initialized, "模型未正確初始化"
            assert trainer.beta is not None, "模型參數未設定"
            assert len(trainer.beta) == self.n_features + 1, "參數維度不正確"
            
            # 計算性能指標
            mae_history = [r['mae'] for r in results]
            avg_mae = np.mean(mae_history)
            final_mae = mae_history[-1]
            mae_improvement = mae_history[0] - mae_history[-1]
            
            test_results = {
                'avg_mae': avg_mae,
                'final_mae': final_mae,
                'mae_improvement': mae_improvement,
                'training_time': training_time,
                'sparsity': results[-1]['sparsity'],
                'model_state': trainer.get_model_state()
            }
            
            print(f"✓ 無正則化訓練完成:")
            print(f"  - 平均 MAE: {avg_mae:.6f}")
            print(f"  - 最終 MAE: {final_mae:.6f}")
            print(f"  - MAE 改善: {mae_improvement:.6f}")
            print(f"  - 訓練時間: {training_time:.2f} 秒")
            print(f"  - 係數非零項: {test_results['sparsity']}")
            
            return True, test_results
            
        except Exception as e:
            print(f"✗ 無正則化訓練測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return False, {}
    
    def test_streaming_training_with_penalty(self) -> Tuple[bool, Dict[str, Any]]:
        """測試帶正則化的串流訓練"""
        print("\n=== 測試串流訓練（帶正則化）===")
        
        try:
            # 建立資料載入器
            loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            start_time = time.time()
            
            # 執行訓練
            trainer, results = streaming_huber_training(
                data_loader=loader,
                n_features=self.n_features,
                n_batch=self.n_batch,
                penalty=True,
                auto_lambda=True,
                tau_estimation='adaptive',
                mode='fast',
                verbose=True
            )
            
            training_time = time.time() - start_time
            
            # 驗證結果
            assert len(results) == self.n_batch, f"結果數量不正確: {len(results)}"
            assert trainer.is_initialized, "模型未正確初始化"
            assert trainer.beta is not None, "模型參數未設定"
            assert len(trainer.beta) == self.n_features + 1, "參數維度不正確"
            
            # 計算性能指標
            mae_history = [r['mae'] for r in results]
            sparsity_history = [r['sparsity'] for r in results]
            avg_mae = np.mean(mae_history)
            final_mae = mae_history[-1]
            final_sparsity = sparsity_history[-1]
            mae_improvement = mae_history[0] - mae_history[-1]
            
            test_results = {
                'avg_mae': avg_mae,
                'final_mae': final_mae,
                'mae_improvement': mae_improvement,
                'training_time': training_time,
                'sparsity': final_sparsity,
                'sparsity_history': sparsity_history,
                'model_state': trainer.get_model_state()
            }
            
            print(f"✓ 正則化訓練完成:")
            print(f"  - 平均 MAE: {avg_mae:.6f}")
            print(f"  - 最終 MAE: {final_mae:.6f}")
            print(f"  - MAE 改善: {mae_improvement:.6f}")
            print(f"  - 訓練時間: {training_time:.2f} 秒")
            print(f"  - 最終係數非零項: {final_sparsity}")
            print(f"  - 係數非零項範圍: [{min(sparsity_history)}, {max(sparsity_history)}]")
            
            return True, test_results
            
        except Exception as e:
            print(f"✗ 正則化訓練測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return False, {}
    
    def test_prediction_and_evaluation(self) -> bool:
        """測試模型預測和評估"""
        print("\n=== 測試模型預測和評估 ===")
        
        try:
            # 訓練模型
            loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            trainer, _ = streaming_huber_training(
                data_loader=loader,
                n_features=self.n_features,
                n_batch=20,  # 較少批次以加快測試
                penalty=True,
                mode='fast',
                verbose=False
            )
            
            # 獲取測試資料
            test_loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            X_test, y_test = test_loader.get_test_data()
            test_loader.close()
            
            if X_test is None or y_test is None:
                print("⚠ 無法載入測試資料，跳過預測測試")
                return True
            
            # 限制測試資料大小以加快測試
            if len(X_test) > 5000:
                X_test = X_test[:5000]
                y_test = y_test[:5000]
            
            # 進行預測
            y_pred = trainer.predict(X_test)
            
            # 計算評估指標
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # 計算相關係數
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            
            print(f"✓ 預測評估完成:")
            print(f"  - 測試樣本數: {len(X_test):,}")
            print(f"  - MAE: {mae:.6f}")
            print(f"  - RMSE: {rmse:.6f}")
            print(f"  - MAPE: {mape:.2f}%")
            print(f"  - 相關係數: {correlation:.4f}")
            print(f"  - 預測值範圍: [{y_pred.min():.1f}, {y_pred.max():.1f}]")
            print(f"  - 真實值範圍: [{y_test.min():.1f}, {y_test.max():.1f}]")
            
            # 基本合理性檢查
            assert np.all(np.isfinite(y_pred)), "預測值包含非有限值"
            assert mae > 0, "MAE 應該大於 0"
            assert correlation > 0, "相關係數應該為正"
            
            return True
            
        except Exception as e:
            print(f"✗ 預測評估測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_persistence(self) -> bool:
        """測試模型狀態的保存和恢復"""
        print("\n=== 測試模型持久化 ===")
        
        try:
            # 訓練一個小模型
            loader = StreamingDataLoaderWithDask(
                self.data_file_path,
                self.batch_size,
                self.train_samples
            )
            
            trainer, _ = streaming_huber_training(
                data_loader=loader,
                n_features=self.n_features,
                n_batch=10,
                penalty=False,
                verbose=False
            )
            
            # 獲取模型狀態
            model_state = trainer.get_model_state()
            
            # 驗證狀態完整性
            required_keys = ['beta', 'J_cumulative', 'tau', 'batch_count', 
                           'mae_history', 'is_initialized', 'standardizer_stats']
            
            for key in required_keys:
                assert key in model_state, f"模型狀態缺少鍵: {key}"
            
            # 驗證關鍵數值
            assert model_state['beta'] is not None, "beta 參數不應為 None"
            assert len(model_state['beta']) == self.n_features + 1, "beta 維度不正確"
            assert model_state['batch_count'] == 10, "批次計數不正確"
            assert model_state['is_initialized'] == True, "初始化狀態不正確"
            assert len(model_state['mae_history']) == 10, "MAE 歷史長度不正確"
            
            print(f"✓ 模型持久化測試通過:")
            print(f"  - Beta 維度: {len(model_state['beta'])}")
            print(f"  - 批次計數: {model_state['batch_count']}")
            print(f"  - Tau 值: {model_state['tau']:.6f}")
            print(f"  - 標準化樣本數: {trainer.standardizer.count:,}")
            
            return True
            
        except Exception as e:
            print(f"✗ 模型持久化測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """運行所有測試"""
        print("="*60)
        print("開始 YearPredictionMSD 整合測試")
        print("="*60)
        
        results = {}
        
        # 1. 驗證資料檔案
        results['data_file_verification'] = self.verify_data_file()
        if not results['data_file_verification']:
            print("資料檔案驗證失敗，停止測試")
            return results
        
        # 2. 測試資料載入器
        results['data_loader'] = self.test_data_loader()
        
        # 3. 測試無正則化訓練
        success, no_penalty_results = self.test_streaming_training_no_penalty()
        results['training_no_penalty'] = success
        
        # 4. 測試正則化訓練
        success, penalty_results = self.test_streaming_training_with_penalty()
        results['training_with_penalty'] = success
        
        # 5. 測試預測和評估
        results['prediction_evaluation'] = self.test_prediction_and_evaluation()
        
        # 6. 測試模型持久化
        results['model_persistence'] = self.test_model_persistence()
        
        # 總結測試結果
        print("\n" + "="*60)
        print("測試結果總結")
        print("="*60)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✓ 通過" if passed else "✗ 失敗"
            print(f"{test_name:.<30} {status}")
        
        print("-"*60)
        print(f"總計: {passed_tests}/{total_tests} 測試通過")
        
        if passed_tests == total_tests:
            print("🎉 所有測試通過！套件功能正常")
        else:
            print("⚠ 部分測試失敗，需要檢查問題")
        
        return results


def main():
    """主函數"""
    # 設定資料檔案路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '..', '..', '..', 'data', 'YearPredictionMSD.csv')
    
    # 檢查相對路徑
    if not os.path.exists(data_file_path):
        # 嘗試絕對路徑
        data_file_path = "/Users/user/Downloads/巨量資料分析期末/實作嘗試v7/data/YearPredictionMSD.csv"
    
    if not os.path.exists(data_file_path):
        print("錯誤: 找不到 YearPredictionMSD.csv 檔案")
        print("請確保資料檔案位於正確的路徑")
        return
    
    # 運行測試
    test_suite = TestYearPredictionMSDIntegration(data_file_path)
    results = test_suite.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()
