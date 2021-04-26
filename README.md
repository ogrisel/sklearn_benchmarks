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

`sklearn_benchmarks` is used through a command line as described below.

So far, the concurrent libraries available are:

- [daal4py](https://github.com/intel/scikit-learn-intelex) from Intel

The estimators available are:

- [KNeighborsClassifier - Brute force and KD Tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

Benchmark and profiling results can be consulted [here](https://mbatoul.github.io/sklearn_benchmarks/).

## Getting Started

In order to setup the environment, you need to have `conda` installed. See instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To get a local copy up and running follow these simple example steps:

```sh
$ git clone https://github.com/mbatoul/sklearn_benchmarks
$ cd sklearn_benchmarks
$ conda env create --file environment.yml
$ conda activate sklbench
$ pip install .
$ sklearn_benchmarks
# or
$ sklbench
```

## Usage

```sh
Usage: sklbench [OPTIONS]

Options:
  --append, --a                   Append benchmark results to existing ones. By default, all 
                                  existing results will be erased before new ones are made.
  --config, --c TEXT              Path to config file. Default is config.yml.
  --profiling, --p [html|json.gz]
                                  Profiling files type. Default is html.
  --estimator, --e TEXT           Estimator to benchmark. By default, all estimators in config file
                                  will be benchmarked.
  --help                          Show this message and exit.
```
