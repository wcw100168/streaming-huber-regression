"""
Streaming Huber Regression Package

一個用於線上/串流 Huber 回歸的 Python 套件，支援自適應正則化和大規模資料處理。
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 延遲導入以避免依賴問題
__all__ = [
    'StreamingHuberModelTrainer',
    'StreamingDataLoaderWithDask', 
    'OnlineStandardizer',
    'streaming_huber_training'
]

def __getattr__(name):
    """延遲導入主要 API"""
    if name == 'StreamingHuberModelTrainer':
        from .core.trainer import StreamingHuberModelTrainer
        return StreamingHuberModelTrainer
    elif name == 'StreamingDataLoaderWithDask':
        from .data.loader import StreamingDataLoaderWithDask
        return StreamingDataLoaderWithDask
    elif name == 'OnlineStandardizer':
        from .core.standardizer import OnlineStandardizer
        return OnlineStandardizer
    elif name == 'streaming_huber_training':
        from .utils.training import streaming_huber_training
        return streaming_huber_training
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 版本相容性檢查
import sys
if sys.version_info < (3, 7):
    raise RuntimeError("This package requires Python 3.7 or later")
