"""
線上標準化器模組

實作線上標準化器，支援批次更新統計量，使用 Welford's online algorithm。
"""

import numpy as np
from typing import Tuple


class OnlineStandardizer:
    """線上標準化器，支援批次更新統計量"""
    
    def __init__(self, n_features: int):
        """
        初始化線上標準化器
        
        Parameters:
        -----------
        n_features : int
            特徵維度
        """
        self.n_features = n_features
        self.reset()
    
    def reset(self) -> None:
        """重置統計量"""
        self.mean = np.zeros(self.n_features)
        self.M2 = np.zeros(self.n_features)  # 用於計算變異數
        self.count = 0
    
    def update(self, X_batch: np.ndarray) -> None:
        """
        使用 Welford's online algorithm 更新統計量
        
        Parameters:
        -----------
        X_batch : np.ndarray
            批次資料，形狀為 (n_samples, n_features)
        """
        for x in X_batch:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        取得當前的均值和標準差
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (均值, 標準差)
        """
        if self.count < 2:
            return self.mean, np.ones(self.n_features)
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(np.maximum(variance, 1e-8))  # 避免除零
        return self.mean, std
    
    def transform(self, X_batch: np.ndarray) -> np.ndarray:
        """
        標準化批次資料
        
        Parameters:
        -----------
        X_batch : np.ndarray
            待標準化的批次資料
            
        Returns:
        --------
        np.ndarray
            標準化後的資料
        """
        mean, std = self.get_stats()
        return (X_batch - mean) / std
