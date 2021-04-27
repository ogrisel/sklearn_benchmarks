import numpy as np
import pandas as pd
import joblib
import importlib
import time
from viztracer import VizTracer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import set_random_state
from sklearn_benchmarks.config import (
    BENCHMARKING_RESULTS_PATH,
    PROFILING_RESULTS_PATH,
    BENCHMARK_SECONDS_BUDGET,
    BENCHMARK_MAX_ITER,
)
from sklearn_benchmarks.utils.misc import gen_data, predict_or_transform


class BenchFuncExecutor:
    """
    Executes a benchmark function (fit, predict or transform)
    """

    @staticmethod
    def run(func, profiling_output_path, profiling_output_extensions, *args):
        # First run with a profiler (not timed)
        with VizTracer(verbose=0) as tracer:
            tracer.start()
            result = func(*args)
            tracer.stop()
            for extension in profiling_output_extensions:
                output_file = f"{profiling_output_path}.{extension}"
                tracer.save(output_file=output_file)

        # Next runs: at most 10 runs or 30 sec
        times = []
        start = time.perf_counter()
        for _ in range(BENCHMARK_MAX_ITER):
            start_ = time.perf_counter()
            result = func(*args)
            end_ = time.perf_counter()
            times.append(end_ - start_)
            if end_ - start > BENCHMARK_SECONDS_BUDGET:
                break
        mean_time, stdev_time = np.mean(times), np.std(times)
        return (result, mean_time, stdev_time)


class Benchmark:
    """
    Runs benchmarks on one estimator for one library, accross potentially multiple datasets
    """

    def __init__(
        self,
        name="",
        estimator="",
        inherit=False,
        metrics=[],
        hyperparameters={},
        datasets=[],
        random_state=None,
        profiling_output_extensions=[],
    ):
        self.name = name
        self.estimator = estimator
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets
        self.random_state = random_state
        self.profiling_output_extensions = profiling_output_extensions

    def _set_lib(self):
        self.lib_ = self.estimator.split(".")[0]

    def _load_estimator_class(self):
        splitted_path = self.estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        return getattr(importlib.import_module(module), class_name)

    def _init_parameters_grid(self):
        hyperparameters = self.hyperparameters
        if not hyperparameters:
            estimator_class = self._load_estimator_class()
            estimator = estimator_class()
            # Parameters grid should have list values
            hyperparameters = {k: [v] for k, v in estimator.__dict__.items()}
        return ParameterGrid(hyperparameters)

    def _load_metrics_functions(self):
        module = importlib.import_module("sklearn.metrics")
        return [getattr(module, m) for m in self.metrics]

    def _update_all_scores(self):
        df = pd.DataFrame(self.results_)
        for score, value in self.best_scores_.items():
            df[score] = value
        self.results_ = df.to_dict()

    def run(self):
        self._set_lib()
        estimator_class = self._load_estimator_class()
        metrics_functions = self._load_metrics_functions()
        parameters_grid = self._init_parameters_grid()
        self.results_ = []
        for dataset in self.datasets:
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = list(reversed(sorted(dataset["n_samples_test"])))
            for ns_train in n_samples_train:
                X, y = gen_data(
                    dataset["sample_generator"],
                    n_samples=ns_train + max(n_samples_test),
                    n_features=n_features,
                    **dataset["params"],
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=ns_train, random_state=self.random_state
                )
                for params in parameters_grid:
                    estimator = estimator_class(**params)
                    # Set random state on all estimators to ensure deterministic results
                    set_random_state(estimator, random_state=self.random_state)
                    bench_func = estimator.fit
                    # Use digests to identify results later in reporting
                    hyperparams_digest = joblib.hash(params)
                    dataset_digest = joblib.hash(dataset)
                    profiling_output_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_fit_{hyperparams_digest}_{dataset_digest}"

                    _, mean, stdev = BenchFuncExecutor.run(
                        bench_func,
                        profiling_output_path,
                        self.profiling_output_extensions,
                        X_train,
                        y_train,
                    )

                    row = dict(
                        estimator=self.name,
                        function=bench_func.__name__,
                        mean=mean,
                        stdev=stdev,
                        n_samples_train=ns_train,
                        n_samples=ns_train,
                        n_features=n_features,
                        hyperparams_digest=hyperparams_digest,
                        dataset_digest=dataset_digest,
                        **params,
                    )

                    n_iter = 1
                    if hasattr(estimator, "n_iter_"):
                        row["n_iter"] = estimator.n_iter_
                        n_iter = estimator.n_iter_

                    row["throughput"] = X_train.nbytes * n_iter / mean / 1e9

                    self.results_.append(row)

                    print(
                        "%s - %s - %s - n_samples: %i - n_features: %i - mean: %6.7f - stdev: %6.7f"
                        % (
                            self.lib_,
                            self.name,
                            "fit",
                            ns_train,
                            n_features,
                            mean,
                            stdev,
                        )
                    )

                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = predict_or_transform(estimator)
                        profiling_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_{bench_func.__name__}_{hyperparams_digest}_{dataset_digest}"

                        (y_pred, mean, stdev,) = BenchFuncExecutor.run(
                            bench_func,
                            profiling_output_path,
                            self.profiling_output_extensions,
                            X_test_,
                        )

                        # Store the scores computed on the biggest dataset
                        if i == 0:
                            self.best_scores_ = {
                                func.__name__: func(y_test_, y_pred)
                                for func in metrics_functions
                            }

                        row = dict(
                            estimator=self.name,
                            function=bench_func.__name__,
                            mean=mean,
                            stdev=stdev,
                            n_samples_train=ns_train,
                            n_samples=ns_test,
                            n_features=n_features,
                            hyperparams_digest=hyperparams_digest,
                            dataset_digest=dataset_digest,
                            **params,
                        )

                        row["throughput"] = X_test.nbytes / mean / 1e9
                        row["latency"] = mean / X_test.shape[0]
                        if hasattr(estimator, "n_iter_"):
                            row["n_iter"] = estimator.n_iter_

                        print(
                            "%s - %s - %s - n_samples: %i - n_features: %i - mean: %6.7f - stdev: %6.7f"
                            % (
                                self.lib_,
                                self.name,
                                bench_func.__name__,
                                ns_test,
                                n_features,
                                mean,
                                stdev,
                            )
                        )
                        self.results_.append(row)
        self._update_all_scores()
        return self

    def to_csv(self):
        csv_path = f"{BENCHMARKING_RESULTS_PATH}/{self.lib_}_{self.name}.csv"
        pd.DataFrame(self.results_).to_csv(
            csv_path,
            mode="w+",
            index=False,
        )
