"""
串流資料載入器模組

實作帶緩衝區的串流資料載入器，避免批次截斷問題。
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


class StreamingDataLoaderWithDask:
    """帶緩衝區的串流資料載入器 - 避免批次截斷問題"""
    
    def __init__(self, file_path: str, batch_size: int, train_samples: int = 500000, buffer_batches: int = 5):
        """
        初始化帶緩衝區的串流載入器
        
        Parameters:
        -----------
        file_path : str
            資料檔案路徑
        batch_size : int
            每批次大小
        train_samples : int
            訓練樣本總數
        buffer_batches : int
            緩衝區批次數量（預載多少個批次）
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.buffer_batches = buffer_batches
        
        # 狀態變數
        self.samples_read_count = 0
        self.current_batch_idx = 0
        self.buffer = []  # 緩衝區
        self.buffer_position = 0  # 緩衝區當前位置
        self.chunk_reader = None  # pandas chunk reader
        self.is_exhausted = False  # 是否已讀完所有資料
        
        # 統計信息
        self.total_memory_usage = 0
        self.buffer_refill_count = 0
        
    def _initialize_chunk_reader(self):
        """初始化 pandas chunk reader"""
        if self.chunk_reader is None:
            try:
                self.chunk_reader = pd.read_csv(
                    self.file_path,
                    header=None,
                    chunksize=self.batch_size,
                    dtype=np.float32,
                    low_memory=True
                )
            except Exception as e:
                raise RuntimeError(f"無法初始化資料讀取器: {e}")
    
    def _estimate_memory_usage(self, data_chunk: pd.DataFrame) -> int:
        """估計資料塊的記憶體使用量"""
        return data_chunk.memory_usage(deep=True).sum()
    
    def _refill_buffer(self) -> bool:
        """填充緩衝區"""
        if self.is_exhausted:
            return False
            
        if self.chunk_reader is None:
            self._initialize_chunk_reader()
        
        new_buffer = []
        total_buffer_memory = 0
        batches_loaded = 0
        
        try:
            while batches_loaded < self.buffer_batches:
                # 檢查是否已達到訓練樣本限制
                if self.samples_read_count >= self.train_samples:
                    self.is_exhausted = True
                    break
                
                try:
                    chunk = next(self.chunk_reader)
                except StopIteration:
                    self.is_exhausted = True
                    break
                
                if chunk.empty:
                    continue
                
                # 檢查是否超過訓練樣本限制
                remaining_samples = self.train_samples - self.samples_read_count
                if len(chunk) > remaining_samples:
                    chunk = chunk.iloc[:remaining_samples]
                    self.is_exhausted = True
                
                # 分離特徵和標籤
                Y_batch = chunk.iloc[:, 0].values.astype(np.float32)
                X_batch = chunk.iloc[:, 1:].values.astype(np.float32)
                
                batch_data = {
                    'X': X_batch,
                    'y': Y_batch,
                    'batch_id': self.current_batch_idx + batches_loaded,
                    'size': len(X_batch)
                }
                
                # 估計記憶體使用量
                batch_memory = self._estimate_memory_usage(chunk)
                total_buffer_memory += batch_memory
                
                new_buffer.append(batch_data)
                self.samples_read_count += len(X_batch)
                batches_loaded += 1
                
                # 如果這是最後一批（被截斷），立即結束
                if len(chunk) < self.batch_size:
                    self.is_exhausted = True
                    break
        
        except Exception as e:
            print(f"載入資料時發生錯誤: {e}")
            return False
        
        # 更新緩衝區
        self.buffer = new_buffer
        self.buffer_position = 0
        self.current_batch_idx += batches_loaded
        self.total_memory_usage = total_buffer_memory
        self.buffer_refill_count += 1

        return len(new_buffer) > 0
    
    def get_batch(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        獲取下一個批次的資料
        
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (特徵矩陣, 標籤向量) 或 (None, None) 如果沒有更多資料
        """
        # 如果緩衝區為空或已用完，嘗試重新填充
        if not self.buffer or self.buffer_position >= len(self.buffer):
            if not self._refill_buffer():
                return None, None
        
        # 從緩衝區獲取批次
        if self.buffer_position < len(self.buffer):
            batch_data = self.buffer[self.buffer_position]
            self.buffer_position += 1
            
            return batch_data['X'], batch_data['y']
        
        return None, None
    
    def get_test_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        獲取測試資料
        
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (測試特徵矩陣, 測試標籤向量) 或 (None, None) 如果載入失敗
        """
        try:
            test_data = pd.read_csv(
                self.file_path,
                header=None,
                skiprows=self.train_samples,
                dtype=np.float32,
                low_memory=True
            )
            
            if test_data.empty:
                print("測試資料為空")
                return None, None
            
            Y_test = test_data.iloc[:, 0].values
            X_test = test_data.iloc[:, 1:].values
            
            print(f"測試資料載入完成: {len(X_test)} 樣本")
            return X_test, Y_test
            
        except Exception as e:
            print(f"讀取測試資料失敗: {e}")
            return None, None
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """
        獲取載入器統計信息
        
        Returns:
        --------
        Dict[str, Any]
            包含載入器狀態的統計信息
        """
        return {
            'samples_read': self.samples_read_count,
            'current_batch_idx': self.current_batch_idx,
            'buffer_size': len(self.buffer),
            'buffer_position': self.buffer_position,
            'buffer_refills': self.buffer_refill_count,
            'total_memory_mb': self.total_memory_usage / (1024*1024),
            'is_exhausted': self.is_exhausted,
            'completion_rate': self.samples_read_count / self.train_samples * 100
        }
    
    def close(self) -> None:
        """關閉資源並清理緩衝區"""
        if self.chunk_reader:
            try:
                self.chunk_reader.close()
            except:
                pass
        
        # 清理緩衝區
        if self.buffer:
            for batch_data in self.buffer:
                del batch_data['X'], batch_data['y']
            self.buffer.clear()
        
        self.chunk_reader = None
        self.buffer = []
    
    def __del__(self):
        """析構函數，確保資源被釋放"""
        self.close()
