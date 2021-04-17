# Benchmark scikit-learn's estimators against other libraries

## Install

```
$ pip install .
```

## Usage

```
Usage: python -m sklearn_benchmarks [OPTIONS]

Options:
  --append, -a
                                  When set, benchmarking and profiling results
                                  will be added to existing results.
  --config, --c <TEXT>
                                  Path to config file. Default is
                                  sklearn_benchmarks/config.yml.
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
