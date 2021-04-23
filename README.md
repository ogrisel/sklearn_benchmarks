<p align="center">
  <a href="https://github.com/mbatoul/sklearn_benchmarks">
    <img src="logo.png" alt="Logo" >
  </a>

  <h3 align="center">scikit-learn benchmarks</h3>

  <p align="center">
    A comparative benchmarking tool for scikit-learn's estimators
    <br />
  </p>
</p>

## Table of content

<ol>
  <li><a href="#about-the-project">About</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#usage">Usage</a></li>
</ol>

## About

`sklearn_benchmarks` is a framework to benchmark `scikit-learn`'s estimators against concurrent implementations. `sklearn_benchmarks` is written in Python.

Benchmarking results can be visualized in the `reporting.ipynb` notebook. These results are automatically deployed to `github-pages`.

`sklearn_benchmarks` is used through a command line as described below. Ultimately the benchmarks should be as simple as doing:

```sh
$ git clone https://github.com/mbatoul/sklearn_benchmarks
$ sklearn_benchmarks --config config.yml
```

So far, the concurrent libraries available are:

- [daal4py](https://github.com/intel/scikit-learn-intelex) from Intel

The estimators available are:

- [KNeighborsClassifier - Brute force](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KNeighborsClassifier - KD Tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

## Getting Started

To get a local copy up and running follow these simple example steps:

## Usage

```sh
Usage: sklearn_benchmarks [OPTIONS]

Options:
  --append, --a                   Append benchmark results to existing ones.
  --config, --c TEXT              Path to config file.
  --profiling, --p [html|json.gz]
                                  Format of profiling results.
  --help                          Show this message and exit.

```
