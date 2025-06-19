"""
求解器模組初始化
"""

from .irls import IRLSSolver
from .lamm import LAMMSolver

__all__ = ['IRLSSolver', 'LAMMSolver']
