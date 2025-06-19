"""
串流 Huber 回歸模型訓練器

統一處理第一批和後續批次的訓練，適合持續學習場景。
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression

from .standardizer import OnlineStandardizer
from ..solvers.lamm import LAMMSolver
from ..solvers.irls import IRLSSolver


class StreamingHuberModelTrainer:
    """串流 Huber 回歸模型訓練器"""
    
    def __init__(self, n_features: int, tau_estimation_method: str = 'initial',
                 tol: float = 1e-6, max_iter: int = 50, penalty: bool = False,
                 auto_lambda: bool = True):
        """
        初始化串流模型訓練器
        
        Parameters:
        -----------
        n_features : int
            特徵維度
        tau_estimation_method : str
            tau 估計方法 ('initial', 'adaptive', 'fixed')
        tol : float
            收斂容差
        max_iter : int
            最大迭代次數
        penalty : bool
            是否使用正則化
        auto_lambda : bool
            是否自動選擇正則化參數
        """
        self.n_features = n_features
        self.p = n_features + 1  # 包含截距項
        self.tau_estimation_method = tau_estimation_method
        self.tol = tol
        self.max_iter = max_iter
        self.penalty = penalty
        self.auto_lambda = auto_lambda
        
        # 初始化組件
        self.standardizer = OnlineStandardizer(n_features)
        
        # 初始化求解器
        if penalty:
            self.solver = LAMMSolver(tol, max_iter)
        else:
            self.solver = IRLSSolver(tol, max_iter)
        
        # 模型狀態
        self.beta = None
        self.J_cumulative = None
        self.tau = None
        self.batch_count = 0
        self.lambda_current = 0
        self.is_initialized = False
        
        # 統計量
        self.mae_history = []
        self.sparsity_history = []
    
    def _estimate_tau(self, X: np.ndarray, y: np.ndarray) -> float:
        """估計 tau 參數"""
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        resid = y - model.predict(X)
        mad = np.median(np.abs(resid - np.median(resid)))
        return 1.345 * mad
    
    def _huber_loss(self, r: np.ndarray, tau: float) -> np.ndarray:
        """Huber 損失函數"""
        abs_r = np.abs(r)
        return np.where(abs_r <= tau, 0.5 * r**2, tau * abs_r - 0.5 * tau**2)
    
    def _huber_psi(self, r: np.ndarray, tau: float) -> np.ndarray:
        """Huber ψ 函數"""
        return np.where(np.abs(r) <= tau, r, tau * np.sign(r))
    
    def _huber_weights(self, r: np.ndarray, tau: float) -> np.ndarray:
        """IRLS 權重"""
        psi = self._huber_psi(r, tau)
        w = np.empty_like(r)
        zero_mask = (r == 0)
        w[zero_mask] = 1.0
        nonzero = ~zero_mask
        w[nonzero] = psi[nonzero] / r[nonzero]
        return w
    
    def _soft_thresholding(self, u: np.ndarray, alpha: float) -> np.ndarray:
        """軟門檻化函數"""
        return np.sign(u) * np.maximum(np.abs(u) - alpha, 0)
    
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
    
    def _compute_J_matrix(self, X: np.ndarray, y: np.ndarray, 
                         beta: np.ndarray, tau: float) -> np.ndarray:
        """計算資訊矩陣 J"""
        n, p = X.shape
        
        if p > 1 and np.log(p) > 1e-10:
            h = n**(-0.5) / np.log(p)
        else:
            h = 0.1
            
        r = y - X @ beta
        hess = self._smooth_huber_hessian(r, tau, h)
        J = np.einsum('i,ij,ik->jk', hess, X, X)
        return -J
    
    def _compute_U_vector(self, X: np.ndarray, y: np.ndarray, 
                         beta: np.ndarray, tau: float) -> np.ndarray:
        """計算分數向量 U"""
        r = y - X @ beta
        psi = self._huber_psi(r, tau)
        return X.T @ psi
    
    def _bic_criterion(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                      beta: np.ndarray, tau: float, N_b: int) -> tuple:
        """計算 BIC 準則"""
        r = y_batch - X_batch @ beta
        huber_loss_sum = np.sum(self._huber_loss(r, tau))
        df = np.sum(np.abs(beta) > 1e-4)
        bic = np.log(huber_loss_sum / len(y_batch)) + df * np.log(N_b) / N_b
        return bic, df
    
    def _select_lambda_bic(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                          beta_init: np.ndarray, tau: float, N_b: int,
                          beta_prev: np.ndarray, J_prev: np.ndarray,
                          lambda_prev: float, N_prev: int,
                          lambda_candidates: Optional[List[float]] = None,
                          mode: str = 'fast') -> tuple:
        """使用 BIC 準則選擇正則化參數"""
        if lambda_candidates is None:
            # 自動生成候選值
            grad_H = self.solver._compute_H_b_gradient(
                X_batch, y_batch, beta_init, tau, N_b,
                beta_prev, J_prev, lambda_prev, N_prev
            )
            lambda_max = np.max(np.abs(grad_H))
            lambda_min = max(1e-4, lambda_max * 1e-4)
            
            if mode == 'fast':
                lambda_candidates = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 8)
            else:
                lambda_candidates = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 15)
        
        best_lambda = 0
        best_bic = np.inf
        best_beta = beta_init.copy()
        consecutive_worse = 0
        
        for i, lam in enumerate(lambda_candidates):
            try:
                beta_candidate, _ = self.solver.solve(
                    X_batch, y_batch, beta_init, tau, N_b,
                    beta_prev, J_prev, lambda_prev, N_prev, lam
                )
                
                bic_val, _ = self._bic_criterion(X_batch, y_batch, beta_candidate, tau, N_b)
                
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_lambda = lam
                    best_beta = beta_candidate.copy()
                    consecutive_worse = 0
                else:
                    consecutive_worse += 1
                
                if consecutive_worse >= 3 and i >= 3:
                    break
                    
            except Exception:
                consecutive_worse += 1
                continue
        
        return best_lambda, best_beta, best_bic
    
    def _renewable_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """執行 Renewable 更新 (用於第2批之後，無正則化)"""
        beta_old = self.beta.copy()
        J_old = self.J_cumulative.copy()
        
        # 內層迭代更新
        beta_r = beta_old.copy()
        U_r = J_old @ (beta_r - beta_old) + self._compute_U_vector(X, y, beta_r, self.tau)
        
        # Warm-up: 固定步長預更新
        alpha0 = 1.0
        beta_next = beta_r - alpha0 * U_r
        U_next = J_old @ (beta_next - beta_old) + self._compute_U_vector(X, y, beta_next, self.tau)
        
        beta_prev, U_prev = beta_r.copy(), U_r.copy()
        beta_r, U_r = beta_next.copy(), U_next.copy()
        
        # BB 步長迭代
        for r in range(2, self.max_iter + 1):
            eta = beta_r - beta_prev
            psi = U_r - U_prev
            raw_omega = (eta.T @ psi) / (psi.T @ psi + 1e-12)
            alpha = np.minimum(raw_omega, 10.0)
            
            beta_next = beta_r - alpha * U_r
            U_next = J_old @ (beta_next - beta_old) + self._compute_U_vector(X, y, beta_next, self.tau)
            
            if np.linalg.norm(beta_next - beta_r) < self.tol:
                beta_r, U_r = beta_next, U_next
                break
                
            beta_prev, U_prev = beta_r.copy(), U_r.copy()
            beta_r, U_r = beta_next.copy(), U_next.copy()
        
        # 更新模型狀態
        self.beta = beta_r.copy()
        self.J_cumulative = J_old + self._compute_J_matrix(X, y, self.beta, self.tau)
    
    def _renewable_update_penalty(self, X: np.ndarray, y: np.ndarray, 
                                mode: str = 'fast') -> None:
        """執行帶正則化的 Renewable 更新"""
        beta_old = self.beta.copy()
        J_old = self.J_cumulative.copy()
        lambda_old = self.lambda_current
        N_b = self.standardizer.count
        N_prev = N_b - len(y)
        
        # 自動選擇正則化參數
        if self.auto_lambda:
            lambda_current, beta_new, _ = self._select_lambda_bic(
                X, y, beta_old, self.tau, N_b,
                beta_old, J_old, lambda_old, N_prev,
                mode=mode
            )
            
            if lambda_current == 0:
                lambda_current = 0.5 * self.tau * np.sqrt(np.log(len(self.beta)) / N_b)
        else:
            lambda_current = 0.5 * self.tau * np.sqrt(np.log(len(self.beta)) / N_b)
            
            # 使用 LAMM 求解
            try:
                beta_new, _ = self.solver.solve(
                    X, y, beta_old, self.tau, N_b,
                    beta_old, J_old, lambda_old, N_prev, lambda_current
                )
            except Exception as e:
                print(f"LAMM 求解失敗於第 {self.batch_count} 批：{e}")
                beta_new = beta_old.copy()
        
        # 更新模型狀態
        self.beta = beta_new.copy()
        self.J_cumulative = J_old + self._compute_J_matrix(X, y, self.beta, self.tau)
        self.lambda_current = lambda_current
        
        sparsity = np.sum(np.abs(beta_new) > 1e-4)
        self.sparsity_history.append(sparsity)
    
    def train_on_batch(self, X_raw: np.ndarray, y: np.ndarray, 
                      penalty: bool = False, mode: str = 'fast', 
                      auto_lambda: bool = True) -> Dict[str, Any]:
        """
        在單一批次上訓練模型
        
        Parameters:
        -----------
        X_raw : np.ndarray
            原始特徵矩陣 (n_samples, n_features)
        y : np.ndarray
            目標向量 (n_samples,)
        penalty : bool
            是否使用正則化
        mode : str
            訓練模式 ('fast' 或 'standard')
        auto_lambda : bool
            是否自動選擇正則化參數
            
        Returns:
        --------
        Dict[str, Any]
            包含訓練結果的字典
        """
        # 更新標準化統計量
        self.standardizer.update(X_raw)
        X_std = self.standardizer.transform(X_raw)
        
        # 添加截距項
        X = np.hstack([X_std, np.ones((X_std.shape[0], 1))])
        
        self.batch_count += 1
        
        if not self.is_initialized:
            # 第一批：初始化模型
            if self.tau_estimation_method == 'initial' or self.tau is None:
                self.tau = self._estimate_tau(X, y)
            
            if penalty:
                # 使用帶 Lasso 的 IRLS
                self.beta = self._estimate_beta_first_batch_lasso(X, y, self.tau)
            else:
                # 使用標準 IRLS
                self.beta, _ = self.solver.solve(X, y, self.tau)
                
            self.J_cumulative = self._compute_J_matrix(X, y, self.beta, self.tau)
            self.is_initialized = True
            
        else:
            # 後續批次：使用 Renewable 更新
            if self.tau_estimation_method == 'adaptive':
                self.tau = self._estimate_tau(X, y)
            
            if penalty:
                self._renewable_update_penalty(X, y, mode=mode)
            else:
                self._renewable_update(X, y)
        
        # 計算當前批次性能
        y_pred = X @ self.beta
        current_mae = np.mean(np.abs(y - y_pred))
        self.mae_history.append(current_mae)
        
        return {
            'batch_id': self.batch_count,
            'mae': current_mae,
            'sparsity': np.sum(np.abs(self.beta) > 1e-4),
            'beta': self.beta.copy(),
            'tau': self.tau,
            'is_first_batch': self.batch_count == 1
        }
    
    def _estimate_beta_first_batch_lasso(self, X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
        """使用 Lasso 估計第一批的 beta"""
        n, p = X.shape
        
        # 初始化
        if self.beta is not None:
            beta = self.beta.copy()
        else:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Lasso 正則化參數
        lam = 0.5 * tau * np.sqrt(np.log(p) / n)
        
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            
            # 計算殘差和權重
            r = y - X @ beta
            w = self._huber_weights(r, tau)
            
            # 解加權最小平方
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWy = X.T @ (W @ y)
            beta_tilde = np.linalg.solve(XtWX, XtWy)
            
            # 軟門檻化（不對截距項進行正則化）
            thresh = lam
            beta = beta_tilde.copy()
            beta[:p-1] = self._soft_thresholding(beta_tilde[:p-1], thresh)
            
            # 檢查收斂
            if np.linalg.norm(beta - beta_old) < self.tol:
                break
        
        return beta
    
    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """
        預測新資料
        
        Parameters:
        -----------
        X_raw : np.ndarray
            原始特徵矩陣
            
        Returns:
        --------
        np.ndarray
            預測結果
        """
        if not self.is_initialized:
            raise ValueError("Model must be trained before prediction")
        
        X_std = self.standardizer.transform(X_raw)
        X = np.hstack([X_std, np.ones((X_std.shape[0], 1))])
        return X @ self.beta
    
    def get_model_state(self) -> Dict[str, Any]:
        """獲取模型當前狀態"""
        return {
            'beta': self.beta.copy() if self.beta is not None else None,
            'J_cumulative': self.J_cumulative.copy() if self.J_cumulative is not None else None,
            'tau': self.tau,
            'batch_count': self.batch_count,
            'mae_history': self.mae_history.copy(),
            'sparsity_history': self.sparsity_history.copy(),
            'is_initialized': self.is_initialized,
            'standardizer_stats': self.standardizer.get_stats()
        }
    
    def reset(self) -> None:
        """重置模型狀態"""
        self.beta = None
        self.J_cumulative = None
        self.tau = None
        self.batch_count = 0
        self.is_initialized = False
        self.mae_history = []
        self.sparsity_history = []
        self.standardizer.reset()
