# Benchmark scikit-learn's estimators against other libraries

## Install

```
$ pip install .
```

## Usage

```
Usage: python -m sklearn_benchmarks [OPTIONS]

Options:
  --append, --a                   Append benchmark results to existing ones.
  --config, --c TEXT              Path to config file.
  --profiling_file_type, --pft TEXT
                                  Profiling files type.
  --help                          Show this message and exit.

```

## List of estimators available

- Against [daal4py](https://github.com/intel/scikit-learn-intelex)
  - [KNeighborsClassifier (brute force)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  - [KNeighborsClassifier (kd tree)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Results

[See here](https://mbatoul.github.io/sklearn_benchmarks/)
