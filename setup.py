import yaml
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("environment.yml", "r") as f:
    environment = yaml.load(f)
    for elem in environment["dependencies"]:
        requirements.append(elem)

setup(
    name="sklearn_benchmarks",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "sklearn_benchmarks = sklearn_benchmarks.__main__:main",
            "sklbench = sklearn_benchmarks.__main__:main",
        ]
    },
    version="1.0.0",
    description="A comparative benchmarking tool for scikit-learn's estimators",
    long_description=long_description,
    py_modules=["sklearn_benchmarks"],
    install_requires=requirements,
)
