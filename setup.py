from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fairness-toolkit",
    version="0.1.0",
    author="Jonathan Tsai",
    author_email="chuan-ching_tsai@rush.edu",
    description="A Python package for evaluating and improving fairness in binary classification models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cctsairush/fairness_toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "jinja2>=2.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ]
    }
)