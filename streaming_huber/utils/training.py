"""
串流訓練便利函數

提供簡化的 API 來執行串流 Huber 回歸訓練。
"""

import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
try:
    from tqdm import tqdm
except ImportError:
    # 如果沒有安裝 tqdm，使用簡單的替代實現
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.current = 0
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                progress = self.current / self.total * 100
                print(f"\rProgress: {progress:.1f}%", end="")
        
        def set_postfix_str(self, s):
            print(f" - {s}", end="")
        
        def close(self):
            print()

from ..core.trainer import StreamingHuberModelTrainer
from ..data.loader import StreamingDataLoaderWithDask


def streaming_huber_training(
    data_loader: StreamingDataLoaderWithDask,
    n_features: int,
    n_batch: Optional[int] = None,
    penalty: bool = False,
    auto_lambda: bool = True,
    tau_estimation: str = 'initial',
    mode: str = 'standard',
    tol: float = 1e-6,
    max_iter: int = 500,
    verbose: bool = True
) -> Tuple[StreamingHuberModelTrainer, List[Dict[str, Any]]]:
    """
    便利函數：執行串流 Huber 回歸訓練
    
    Parameters:
    -----------
    data_loader : StreamingDataLoaderWithDask
        資料載入器
    n_features : int
        特徵維度
    n_batch : Optional[int]
        最大批次數量
    penalty : bool
        是否使用正則化
    auto_lambda : bool
        是否自動選擇正則化參數
    tau_estimation : str
        tau 估計方法
    mode : str
        訓練模式 ('fast' 或 'standard')
    tol : float
        收斂容差
    max_iter : int
        最大迭代次數
    verbose : bool
        是否顯示訓練進度
        
    Returns:
    --------
    Tuple[StreamingHuberModelTrainer, List[Dict[str, Any]]]
        訓練完成的模型和訓練結果
    """
    if verbose:
        print("=== 開始串流 Huber 模型訓練 ===")
        print(f"特徵維度: {n_features}")
        print(f"批次大小: {data_loader.batch_size}")
        print(f"正則化: {penalty}")
    
    # 初始化訓練器
    trainer = StreamingHuberModelTrainer(
        n_features=n_features,
        tau_estimation_method=tau_estimation,
        penalty=penalty,
        auto_lambda=auto_lambda,
        tol=tol,
        max_iter=max_iter
    )
    
    # 估計總批次數
    if n_batch is not None:
        total_batches = n_batch
    else:
        total_batches = data_loader.train_samples // data_loader.batch_size
        if data_loader.train_samples % data_loader.batch_size != 0:
            total_batches += 1
    
    # 初始化進度條
    progress_bar = None
    if verbose:
        progress_bar = tqdm(
            total=total_batches,
            desc="Training",
            unit="batch",
            ncols=100
        )
    
    # 訓練循環
    training_results = []
    batch_count = 0
    start_time = time.time()
    
    try:
        while True:
            if n_batch is not None and batch_count >= n_batch:
                break
                
            X_batch, y_batch = data_loader.get_batch()
            if X_batch is None:
                break
            
            # 訓練模型
            result = trainer.train_on_batch(
                X_batch, y_batch, 
                penalty=penalty, 
                mode=mode, 
                auto_lambda=auto_lambda
            )
            training_results.append(result)
            batch_count += 1
            
            if progress_bar:
                progress_bar.set_postfix_str(f"MAE: {result['mae']:.6f}")
                progress_bar.update(1)
                
    finally:
        if progress_bar:
            progress_bar.close()
        data_loader.close()
    
    training_time = time.time() - start_time
    
    if verbose:
        avg_mae = sum(r['mae'] for r in training_results) / len(training_results)
        print(f"\n=== 訓練完成 ===")
        print(f"總批次數: {batch_count}")
        print(f"平均 MAE: {avg_mae:.6f}")
        print(f"訓練時間: {training_time:.2f} 秒")
    
    return trainer, training_results
