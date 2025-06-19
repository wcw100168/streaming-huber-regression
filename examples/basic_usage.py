"""
基本使用範例

展示如何使用串流 Huber 回歸套件進行基本的模型訓練和預測。
"""

import os
import sys
import numpy as np

# 添加套件路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from streaming_huber import (
    StreamingDataLoaderWithDask,
    StreamingHuberModelTrainer,
    streaming_huber_training
)


def basic_usage_example():
    """基本使用範例"""
    print("="*60)
    print("串流 Huber 回歸基本使用範例")
    print("="*60)
    
    # 設定資料檔案路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '..', 'data', 'YearPredictionMSD.csv')
    
    # 檢查資料檔案是否存在
    if not os.path.exists(data_file_path):
        data_file_path = "/Users/user/Downloads/巨量資料分析期末/實作嘗試v7/data/YearPredictionMSD.csv"
    
    if not os.path.exists(data_file_path):
        print("錯誤: 找不到 YearPredictionMSD.csv 檔案")
        print("請確保資料檔案位於正確的路徑")
        return
    
    print(f"使用資料檔案: {data_file_path}")
    
    # 設定參數
    n_features = 90  # YearPredictionMSD 有 90 個特徵
    batch_size = 1000
    train_samples = 10000  # 使用較少樣本以加快範例運行
    n_batch = 10
    
    print(f"參數設定:")
    print(f"  - 特徵維度: {n_features}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 訓練樣本數: {train_samples:,}")
    print(f"  - 批次數量: {n_batch}")
    
    # 方法 1: 使用便利函數
    print("\n" + "="*40)
    print("方法 1: 使用便利函數")
    print("="*40)
    
    try:
        # 建立資料載入器
        data_loader = StreamingDataLoaderWithDask(
            data_file_path,
            batch_size,
            train_samples
        )
        
        # 使用便利函數進行訓練
        trainer, results = streaming_huber_training(
            data_loader=data_loader,
            n_features=n_features,
            n_batch=n_batch,
            penalty=False,  # 不使用正則化
            tau_estimation='adaptive',
            verbose=True
        )
        
        # 顯示訓練結果
        avg_mae = np.mean([r['mae'] for r in results])
        print(f"\n訓練完成:")
        print(f"  - 平均 MAE: {avg_mae:.6f}")
        print(f"  - 最終 MAE: {results[-1]['mae']:.6f}")
        print(f"  - 係數非零項: {results[-1]['sparsity']}")
        
        # 進行預測
        test_loader = StreamingDataLoaderWithDask(
            data_file_path,
            batch_size,
            train_samples
        )
        X_test, y_test = test_loader.get_test_data()
        test_loader.close()
        
        if X_test is not None and y_test is not None:
            # 限制測試樣本數以加快範例
            if len(X_test) > 1000:
                X_test = X_test[:1000]
                y_test = y_test[:1000]
            
            y_pred = trainer.predict(X_test)
            test_mae = np.mean(np.abs(y_test - y_pred))
            
            print(f"\n預測結果:")
            print(f"  - 測試 MAE: {test_mae:.6f}")
            print(f"  - 預測值範圍: [{y_pred.min():.1f}, {y_pred.max():.1f}]")
            print(f"  - 真實值範圍: [{y_test.min():.1f}, {y_test.max():.1f}]")
        
        print("✓ 方法 1 完成")
        
    except Exception as e:
        print(f"✗ 方法 1 失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法 2: 手動使用各個組件
    print("\n" + "="*40)
    print("方法 2: 手動使用各個組件")
    print("="*40)
    
    try:
        # 建立資料載入器
        data_loader = StreamingDataLoaderWithDask(
            data_file_path,
            batch_size,
            train_samples
        )
        
        # 建立訓練器
        trainer = StreamingHuberModelTrainer(
            n_features=n_features,
            tau_estimation_method='initial',
            penalty=True,  # 使用正則化
            auto_lambda=True
        )
        
        print("開始手動訓練...")
        
        # 手動訓練循環
        training_results = []
        for batch_idx in range(n_batch):
            X_batch, y_batch = data_loader.get_batch()
            if X_batch is None:
                break
            
            # 訓練單個批次
            result = trainer.train_on_batch(
                X_batch, y_batch,
                penalty=True,
                mode='fast',
                auto_lambda=True
            )
            
            training_results.append(result)
            
            if batch_idx % 2 == 0:  # 每 2 個批次顯示進度
                print(f"  批次 {result['batch_id']}: MAE = {result['mae']:.6f}, "
                      f"係數非零項 = {result['sparsity']}")
        
        data_loader.close()
        
        # 顯示最終結果
        if training_results:
            final_mae = training_results[-1]['mae']
            final_sparsity = training_results[-1]['sparsity']
            mae_improvement = training_results[0]['mae'] - training_results[-1]['mae']
            
            print(f"\n手動訓練完成:")
            print(f"  - 最終 MAE: {final_mae:.6f}")
            print(f"  - MAE 改善: {mae_improvement:.6f}")
            print(f"  - 最終係數非零項: {final_sparsity}")
            
            # 顯示模型狀態
            model_state = trainer.get_model_state()
            print(f"  - 處理批次數: {model_state['batch_count']}")
            print(f"  - 累積樣本數: {trainer.standardizer.count:,}")
            print(f"  - Tau 值: {model_state['tau']:.6f}")
        
        print("✓ 方法 2 完成")
        
    except Exception as e:
        print(f"✗ 方法 2 失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("基本使用範例完成")
    print("="*60)


def advanced_usage_example():
    """進階使用範例：比較不同配置"""
    print("\n" + "="*60)
    print("進階使用範例：比較不同配置")
    print("="*60)
    
    # 設定資料檔案路徑（同上）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '..', 'data', 'YearPredictionMSD.csv')
    
    if not os.path.exists(data_file_path):
        data_file_path = "/Users/user/Downloads/巨量資料分析期末/實作嘗試v7/data/YearPredictionMSD.csv"
    
    if not os.path.exists(data_file_path):
        print("錯誤: 找不到資料檔案，跳過進階範例")
        return
    
    # 設定參數
    n_features = 90
    batch_size = 800
    train_samples = 8000
    n_batch = 10
    
    # 測試不同配置
    configs = [
        {'name': '無正則化', 'penalty': False, 'tau_estimation': 'initial'},
        {'name': '自動正則化', 'penalty': True, 'auto_lambda': True, 'tau_estimation': 'adaptive'},
        {'name': '快速模式', 'penalty': True, 'auto_lambda': True, 'mode': 'fast'},
    ]
    
    results_comparison = []
    
    for config in configs:
        print(f"\n測試配置: {config['name']}")
        print("-" * 30)
        
        try:
            # 建立資料載入器
            data_loader = StreamingDataLoaderWithDask(
                data_file_path,
                batch_size,
                train_samples
            )
            
            # 執行訓練
            trainer, results = streaming_huber_training(
                data_loader=data_loader,
                n_features=n_features,
                n_batch=n_batch,
                penalty=config.get('penalty', False),
                auto_lambda=config.get('auto_lambda', True),
                tau_estimation=config.get('tau_estimation', 'initial'),
                mode=config.get('mode', 'standard'),
                verbose=False  # 減少輸出
            )
            
            # 計算統計量
            mae_history = [r['mae'] for r in results]
            avg_mae = np.mean(mae_history)
            final_mae = mae_history[-1]
            sparsity = results[-1]['sparsity']
            
            config_result = {
                'name': config['name'],
                'avg_mae': avg_mae,
                'final_mae': final_mae,
                'sparsity': sparsity,
                'config': config
            }
            
            results_comparison.append(config_result)
            
            print(f"  平均 MAE: {avg_mae:.6f}")
            print(f"  最終 MAE: {final_mae:.6f}")
            print(f"  係數非零項: {sparsity}")
            
        except Exception as e:
            print(f"  配置失敗: {e}")
    
    # 比較結果
    if results_comparison:
        print("\n" + "="*50)
        print("配置比較結果")
        print("="*50)
        
        # 按最終 MAE 排序
        results_comparison.sort(key=lambda x: x['final_mae'])
        
        print(f"{'配置':<15} {'平均MAE':<12} {'最終MAE':<12} {'係數非零項':<10}")
        print("-" * 50)
        
        for result in results_comparison:
            print(f"{result['name']:<15} "
                  f"{result['avg_mae']:<12.6f} "
                  f"{result['final_mae']:<12.6f} "
                  f"{result['sparsity']:<10}")
        
        best_config = results_comparison[0]
        print(f"\n最佳配置: {best_config['name']} (最終 MAE: {best_config['final_mae']:.6f})")
    
    print("\n進階使用範例完成")


if __name__ == "__main__":
    # 運行基本範例
    basic_usage_example()
    
    # 運行進階範例
    advanced_usage_example()
