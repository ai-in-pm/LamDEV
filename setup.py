from setuptools import setup, find_packages

setup(
    name="lam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    python_requires=">=3.10",
)
