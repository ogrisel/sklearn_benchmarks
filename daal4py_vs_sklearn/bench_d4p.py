import time
import importlib
import numpy as np
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import daal4py.sklearn as d4p


def _gen_data_regression(n_samples=1000, n_features=10, random_state=42):
    """Wrapper for sklearn make_regression"""
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           random_state=random_state)
    return X, y


def _gen_data_blobs(n_samples=1000, n_features=10, random_state=42, centers=None):
    """Wrapper for sklearn make_blobs"""
    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=centers,
                      random_state=random_state)
    return X, y


def _gen_data_classification(n_samples=1000, n_features=10, random_state=42, n_classes=2):
    """Wrapper for sklearn make_blobs"""
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_classes=n_classes,
                               random_state=random_state)
    return X, y


_data_generators = {
    'blobs': _gen_data_blobs,
    'classification': _gen_data_classification,
    'regression': _gen_data_regression,
}


def all_datasets():
    return _data_generators


def gen_data(dataset_name, n_samples=1000, n_features=10, random_state=42, **kwargs):
    """Returns a tuple of data from the specified generator."""
    data = _data_generators[dataset_name](
        n_samples, n_features, random_state, **kwargs)

    return data


class BenchmarkTimer:
    """Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:
    .. code-block:: python
        timer = BenchmarkTimer(rep=5)
        for _ in timer.benchmark_runs():
            ... do something ...
        print(np.min(timer.timings))
    """

    def __init__(self, reps=1):
        self.reps = reps
        self.timings = []

    def benchmark_runs(self):
        for r in range(self.reps):
            t0 = time.time()
            yield r
            t1 = time.time()
            self.timings.append(t1 - t0)


class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""

    def __init__(self, bench_rows, bench_dims, dataset_name, n_reps=1):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.n_reps = n_reps

    def _run_one_size(self, algo, n_samples, n_features, param_overrides={}, dataset_param_overrides={}):

        data = gen_data(self.dataset_name, n_samples, n_features, **dataset_param_overrides)

        # sklearn
        algo.reload()
        skl_timer = BenchmarkTimer(self.n_reps)
        for rep in skl_timer.benchmark_runs():
            algo.run(data, **param_overrides)
        skl_elapsed = np.min(skl_timer.timings)

        # patched sklearn
        d4p.patch_sklearn()
        algo.reload()
        d4p_timer = BenchmarkTimer(self.n_reps)
        for rep in d4p_timer.benchmark_runs():
            algo.run(data, **param_overrides)
        d4p_elapsed = np.min(d4p_timer.timings)
        d4p.unpatch_sklearn()

        speedup = skl_elapsed / d4p_elapsed

        print(f"{algo.name} (n_samples={n_samples}, n_features={n_features})"
              f" [skl={skl_elapsed}, d4p={d4p_elapsed} speedup={speedup}]")

        return dict(
            skl_time=skl_elapsed,
            d4p_time=d4p_elapsed,
            speedup=speedup,
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **dataset_param_overrides
        )

    def run(self, algo, param_overrides={}, dataset_param_overrides={}):

        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                all_results.append(
                    self._run_one_size(algo, ns, nf, param_overrides, dataset_param_overrides))
        return all_results


def fit(estimator, X, y=None):
    estimator.fit(X, y)


def fit_kneighbors(estimator, X, y=None):
    estimator.fit(X, y)
    estimator.kneighbors(X)


class Algorithm:
    def __init__(self, algo_module, algo_class, args={}, name=None, bench_func=fit, accepts_labels=True):
        self.name = name
        self.accepts_labels = accepts_labels
        self.algo_module = algo_module
        self.algo_class = algo_class
        self.args = args
        self.bench_func = bench_func

    def reload(self):
        self.algo_module_ = importlib.import_module(self.algo_module)
        self.algo_class_ = getattr(self.algo_module_, self.algo_class)

    def run(self, data, **override_args):
        all_args = {**self.args, **override_args}

        estimator = self.algo_class_(**all_args)
        if self.accepts_labels:
            self.bench_func(estimator, data[0], data[1])
        else:
            self.bench_func(estimator, data[0])
        return estimator


def all_algorithms():
    """Returns all defined AlgorithmPair objects"""
    algorithms = [
        Algorithm(
            algo_module="sklearn.cluster",
            algo_class="KMeans",
            args=dict(init="k-means++", n_clusters=8, max_iter=30, n_init=1, tol=1e-16),
            name="KMeans-random",
            accepts_labels=False,
        ),
        Algorithm(
            algo_module="sklearn.cluster",
            algo_class="KMeans",
            args=dict(init="random", n_clusters=8, max_iter=30, n_init=1, tol=1e-16),
            name="KMeans-kmeans++",
            accepts_labels=False,
        ),
        Algorithm(
            algo_module="sklearn.decomposition",
            algo_class="PCA",
            args=dict(n_components=10),
            name="PCA",
            accepts_labels=False,
        ),
        Algorithm(
            algo_module="sklearn.neighbors",
            algo_class="NearestNeighbors",
            args=dict(n_neighbors=1024, algorithm="brute", n_jobs=-1),
            name="NearestNeighbors",
            accepts_labels=False,
            bench_func=fit_kneighbors,
        ),
        Algorithm(
            algo_module="sklearn.cluster",
            algo_class="DBSCAN",
            args=dict(eps=3, min_samples=2, algorithm="brute"),
            name="DBSCAN",
            accepts_labels=False,
        ),
        Algorithm(
            algo_module="sklearn.linear_model",
            algo_class="LinearRegression",
            args={},
            name="LinearRegression",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.linear_model",
            algo_class="ElasticNet",
            args={"alpha": 0.1, "l1_ratio": 0.5},
            name="ElasticNet",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.linear_model",
            algo_class="Lasso",
            args={},
            name="Lasso",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.linear_model",
            algo_class="Ridge",
            args={},
            name="Ridge",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.linear_model",
            algo_class="LogisticRegression",
            args=dict(),
            name="LogisticRegression",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.ensemble",
            algo_class="RandomForestClassifier",
            args={"max_features": 1.0, "n_estimators": 10},
            name="RandomForestClassifier",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.ensemble",
            algo_class="RandomForestRegressor",
            args={"max_features": 1.0, "n_estimators": 10},
            name="RandomForestRegressor",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.manifold",
            algo_class="TSNE",
            args=dict(),
            name="TSNE",
            accepts_labels=False,
        ),
        Algorithm(
            algo_module="sklearn.svm",
            algo_class="SVC",
            args={"kernel": "rbf"},
            name="SVC-RBF",
            accepts_labels=True,
        ),
        Algorithm(
            algo_module="sklearn.svm",
            algo_class="SVC",
            args={"kernel": "linear"},
            name="SVC-Linear",
            accepts_labels=True,
        )
    ]

    return algorithms


def algorithm_by_name(name):
    """Returns the algorithm pair with the name 'name' (case-insensitive)"""
    algos = all_algorithms()
    return next((a for a in algos if a.name.lower() == name.lower()), None)

