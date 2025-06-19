"""
IRLS (迭代重新加權最小平方) 求解器

實作用於 Huber 迴歸的 IRLS 演算法。
"""

import numpy as np
from typing import Tuple, Dict, Any


class IRLSSolver:
    """IRLS 求解器，用於求解 Huber 迴歸問題"""
    
    def __init__(self, tol: float = 1e-6, max_iter: int = 50):
        """
        初始化 IRLS 求解器
        
        Parameters:
        -----------
        tol : float
            收斂容差
        max_iter : int
            最大迭代次數
        """
        self.tol = tol
        self.max_iter = max_iter
    
    def _huber_psi(self, r: np.ndarray, tau: float) -> np.ndarray:
        """Huber ψ 函數"""
        return np.where(np.abs(r) <= tau, r, tau * np.sign(r))
    
    def _huber_weights(self, r: np.ndarray, tau: float) -> np.ndarray:
        """IRLS 權重計算"""
        psi = self._huber_psi(r, tau)
        w = np.empty_like(r)
        zero_mask = (r == 0)
        w[zero_mask] = 1.0
        nonzero = ~zero_mask
        w[nonzero] = psi[nonzero] / r[nonzero]
        return w
    
    def solve(self, X: np.ndarray, y: np.ndarray, tau: float, 
              beta_init: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用 IRLS 求解 Huber 迴歸
        
        Parameters:
        -----------
        X : np.ndarray
            設計矩陣，形狀為 (n_samples, n_features)
        y : np.ndarray
            響應向量，形狀為 (n_samples,)
        tau : float
            Huber 參數
        beta_init : np.ndarray, optional
            初始參數估計，如果為 None 則使用 OLS
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            (最終參數估計, 求解統計信息)
        """
        n, p = X.shape
        
        # 初始化：如果已有估計，使用之；否則使用 OLS
        if beta_init is not None:
            beta = beta_init.copy()
        else:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        history = {
            'objective': [],
            'beta_norm': [],
            'residual_norm': []
        }
        
        for k in range(self.max_iter):
            # 計算殘差和權重
            r = y - X @ beta
            w = self._huber_weights(r, tau)
            W = np.diag(w)
            
            # 解加權最小平方問題
            XtWX = X.T @ W @ X
            XtWy = X.T @ (W @ y)
            
            try:
                beta_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                # 如果矩陣奇異，使用偽逆
                beta_new = np.linalg.pinv(XtWX) @ XtWy
            
            # 記錄歷史
            residual_norm = np.linalg.norm(r)
            history['objective'].append(residual_norm)
            history['beta_norm'].append(np.linalg.norm(beta_new))
            history['residual_norm'].append(residual_norm)
            
            # 檢查收斂
            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
                
            beta = beta_new
        
        return beta, {
            'iterations': k + 1,
            'converged': k < self.max_iter - 1,
            'final_residual_norm': history['residual_norm'][-1],
            'history': history
        }
