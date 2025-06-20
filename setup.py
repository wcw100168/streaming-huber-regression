"""
套件安裝配置檔案
"""

from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8') if (HERE / "README.md").exists() else ""

# 讀取版本號
def get_version():
    """從 __init__.py 讀取版本號"""
    with open('streaming_huber/__init__.py', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name="streaming-huber-regression",
    version=get_version(),
    description="一個用於線上/串流 Huber 回歸的 Python 套件，支援自適應正則化和大規模資料處理",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/streaming-huber-regression",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=2.3.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    python_requires=">=3.7",
    package_data={
        "streaming_huber": ["*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/streaming-huber-regression/issues",
        "Source": "https://github.com/yourusername/streaming-huber-regression",
    },
)
