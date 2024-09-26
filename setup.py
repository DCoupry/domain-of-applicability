from setuptools import setup, find_packages

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doa",
    version="0.1.0",
    author="Damien Coupry",
    author_email="damien.coupry+github@pm.me",
    description="A package for anomaly detection and domain of applicability estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DCoupry/doa",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
)
