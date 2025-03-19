from setuptools import setup, find_packages

setup(
    name="pytorch_lns",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "tabulate>=0.9.0",
    ],
    author="AbhiRam162105",
    author_email="your.email@example.com",
    description="Logarithmic Number System (LNS) implementation for PyTorch",
    keywords="pytorch, lns, deep-learning",
    python_requires=">=3.8",
)