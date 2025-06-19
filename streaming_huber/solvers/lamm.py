"""
LAMM (Local Adaptive MM) 求解器

實作用於帶正則化的 Huber 迴歸的 LAMM 演算法。
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class LAMMSolver:
    """LAMM 求解器，用於求解帶 L1 正則化的 Huber 迴歸問題"""
    
    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        """
        初始化 LAMM 求解器
        
        Parameters:
        -----------
        tol : float
            收斂容差
        max_iter : int
            最大迭代次數
        """
        self.tol = tol
        self.max_iter = max_iter
    
    def _huber_loss(self, r: np.ndarray, tau: float) -> np.ndarray:
        """Huber 損失函數"""
        abs_r = np.abs(r)
        return np.where(abs_r <= tau, 0.5 * r**2, tau * abs_r - 0.5 * tau**2)
    
    def _huber_psi(self, r: np.ndarray, tau: float) -> np.ndarray:
        """Huber ψ 函數"""
        return np.where(np.abs(r) <= tau, r, tau * np.sign(r))
    
    def _smooth_huber_hessian(self, r: np.ndarray, tau: float, h: float) -> np.ndarray:
        """平滑 Huber 二階導數"""
        abs_r = np.abs(r)
        hess = np.zeros_like(r)
        
        in_core = abs_r <= tau - h
        hess[in_core] = 1
        
        left = (r >= -tau - h) & (r <= -tau + h)
        hess[left] = 0.5 + (r[left] + tau) / (2 * h)
        
        right = (r >= tau - h) & (r <= tau + h)
        hess[right] = 0.5 - (r[right] - tau) / (2 * h)
        
        return hess
    
    def _soft_thresholding(self, u: np.ndarray, alpha: float) -> np.ndarray:
        """軟門檻化函數"""
        return np.sign(u) * np.maximum(np.abs(u) - alpha, 0)
    
    def _compute_J_matrix(self, X: np.ndarray, y: np.ndarray, 
                         beta: np.ndarray, tau: float) -> np.ndarray:
        """計算資訊矩陣 J"""
        n, p = X.shape
        
        # 避免 log(p) = 0 的情況
        if p > 1 and np.log(p) > 1e-10:
            h = n**(-0.5) / np.log(p)
        else:
            h = 0.1
            
        r = y - X @ beta
        hess = self._smooth_huber_hessian(r, tau, h)
        
        # 向量化計算
        J = np.einsum('i,ij,ik->jk', hess, X, X)
        return -J
    
    def _compute_U_vector(self, X: np.ndarray, y: np.ndarray, 
                         beta: np.ndarray, tau: float) -> np.ndarray:
        """計算分數向量 U"""
        r = y - X @ beta
        psi = self._huber_psi(r, tau)
        return X.T @ psi
    
    def _compute_H_b(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                    beta: np.ndarray, tau: float, N_b: int,
                    beta_prev: np.ndarray, J_prev: np.ndarray, 
                    lambda_prev: float, N_prev: int) -> float:
        """計算 H_b(β) 函數"""
        # Huber 損失項
        r = y_batch - X_batch @ beta
        huber_loss_sum = np.sum(self._huber_loss(r, tau))
        huber_term = huber_loss_sum / N_b
        
        # 二次項
        beta_diff = beta - beta_prev
        quadratic_term = -0.5 / N_b * (beta_diff.T @ J_prev @ beta_diff)
        
        # 線性項
        linear_term = 0
        if N_prev > 0 and lambda_prev > 0:
            sgn_beta_prev = np.sign(beta_prev)
            linear_term = -(N_prev / N_b) * lambda_prev * (beta_diff.T @ sgn_beta_prev)
        
        return huber_term + quadratic_term + linear_term
    
    def _compute_H_b_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray,
                             beta: np.ndarray, tau: float, N_b: int,
                             beta_prev: np.ndarray, J_prev: np.ndarray,
                             lambda_prev: float, N_prev: int) -> np.ndarray:
        """計算 H_b(β) 的梯度"""
        # Huber 損失梯度項
        huber_grad = -self._compute_U_vector(X_batch, y_batch, beta, tau) / N_b
        
        # 二次項梯度
        quadratic_grad = -J_prev @ (beta - beta_prev) / N_b
        
        # 線性項梯度
        linear_grad = np.zeros_like(beta)
        if N_prev > 0 and lambda_prev > 0:
            sgn_beta_prev = np.sign(beta_prev)
            linear_grad = -(N_prev / N_b) * lambda_prev * sgn_beta_prev
        
        return huber_grad + quadratic_grad + linear_grad
    
    def solve(self, X_batch: np.ndarray, y_batch: np.ndarray, 
             beta_init: np.ndarray, tau: float, N_b: int,
             beta_prev: np.ndarray, J_prev: np.ndarray,
             lambda_prev: float, N_prev: int, lambda_current: float,
             phi_init: float = 1e-6, omega: float = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用 LAMM 演算法求解
        
        Parameters:
        -----------
        X_batch : np.ndarray
            當前批次的設計矩陣
        y_batch : np.ndarray
            當前批次的響應向量
        beta_init : np.ndarray
            初始參數估計
        tau : float
            Huber 參數
        N_b : int
            累積樣本數
        beta_prev : np.ndarray
            前一批次的參數估計
        J_prev : np.ndarray
            前一批次的資訊矩陣
        lambda_prev : float
            前一批次的正則化參數
        N_prev : int
            前一批次的累積樣本數
        lambda_current : float
            當前批次的正則化參數
        phi_init : float
            初始 phi 值
        omega : float
            phi 更新倍數
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            (最終參數估計, 求解統計信息)
        """
        beta_current = beta_init.copy()
        phi = phi_init
        history = {'objective': [], 'phi': [], 'beta_norm': []}
        
        for iteration in range(self.max_iter):
            # 計算當前梯度
            grad_H = self._compute_H_b_gradient(
                X_batch, y_batch, beta_current, tau, N_b,
                beta_prev, J_prev, lambda_prev, N_prev
            )
            
            # 計算當前目標函數值
            H_current = self._compute_H_b(
                X_batch, y_batch, beta_current, tau, N_b,
                beta_prev, J_prev, lambda_prev, N_prev
            )
            
            # 內層循環：調整 phi 直到滿足增上界條件
            phi_adjusted = False
            max_phi_iter = 10
            current_phi = phi
            
            for phi_iter in range(max_phi_iter):
                # 計算候選解（軟門檻化）
                u = beta_current - grad_H / current_phi
                beta_candidate = self._soft_thresholding(u, lambda_current / current_phi)
                
                # 檢查增上界條件
                H_candidate = self._compute_H_b(
                    X_batch, y_batch, beta_candidate, tau, N_b,
                    beta_prev, J_prev, lambda_prev, N_prev
                )
                
                # 計算代理函數值
                beta_diff = beta_candidate - beta_current
                surrogate_val = (H_current + grad_H.T @ beta_diff + 
                               0.5 * current_phi * np.linalg.norm(beta_diff)**2)
                
                if surrogate_val >= H_candidate - 1e-10:  # 增上界條件滿足
                    phi_adjusted = True
                    phi = current_phi
                    break
                else:
                    current_phi *= omega
            
            if not phi_adjusted:
                break
            
            # 更新參數
            beta_new = beta_candidate.copy()
            
            # 記錄歷史
            current_obj = H_candidate + lambda_current * np.linalg.norm(beta_new, 1)
            history['objective'].append(current_obj)
            history['phi'].append(phi)
            history['beta_norm'].append(np.linalg.norm(beta_new))
            
            # 檢查收斂
            param_change = np.linalg.norm(beta_new - beta_current)
            if param_change <= self.tol:
                break
            
            beta_current = beta_new
        
        return beta_current, {
            'iterations': iteration + 1,
            'converged': iteration < self.max_iter - 1,
            'final_phi': phi,
            'history': history
        }
