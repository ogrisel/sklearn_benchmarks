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

- KNeighborsClassifier (brute force)
- KNeighborsClassifier (kd tree)
- KMeans

## List of libraries available

- [daal4py](https://github.com/intel/scikit-learn-intelex)

## Results

[See results](https://mbatoul.github.io/sklearn_benchmarks/)
