import importlib
import random
import time
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils._testing import set_random_state
from viztracer import VizTracer

from sklearn_benchmarks.config import (
    BENCHMARK_MAX_ITER,
    FUNC_TIME_BUDGET,
    BENCHMARKING_RESULTS_PATH,
    PROFILING_RESULTS_PATH,
    BENCHMARK_TIME_BUDGET,
)
from sklearn_benchmarks.utils.misc import gen_data, predict_or_transform


class BenchFuncExecutor:
    """
    Executes a benchmark function (fit, predict or transform)
    """

    def run(
        self,
        func,
        estimator,
        profiling_output_path,
        profiling_output_extensions,
        X,
        y=None,
        max_iter=BENCHMARK_MAX_ITER,
        **kwargs,
    ):
        # First run with a profiler (not timed)
        # with VizTracer(verbose=0) as tracer:
        #     tracer.start()
        #     if y is not None:
        #         func(X, y, **kwargs)
        #     else:
        #         func(X, **kwargs)
        #     tracer.stop()
        #     for extension in profiling_output_extensions:
        #         output_file = f"{profiling_output_path}.{extension}"
        #         tracer.save(output_file=output_file)

        # Next runs: at most 10 runs or 30 sec
        times = []
        start = time.perf_counter()
        for _ in range(max_iter):
            start = time.perf_counter()

            if y is not None:
                self.func_result_ = func(X, y, **kwargs)
            else:
                self.func_result_ = func(X, **kwargs)

            end = time.perf_counter()
            times.append(end - start)

            if end - start > FUNC_TIME_BUDGET:
                break

        benchmark_info = {}
        mean = np.mean(times)

        n_iter = None
        if hasattr(estimator, "n_iter_"):
            benchmark_info["n_iter"] = estimator.n_iter_
            n_iter = estimator.n_iter_
        n_iter = 1 if n_iter is None else n_iter

        benchmark_info["mean_time"] = mean
        benchmark_info["stdev_time"] = np.std(times)
        benchmark_info["throughput"] = X.nbytes * n_iter / mean / 1e9
        benchmark_info["latency"] = mean / X.shape[0]

        return benchmark_info


class Benchmark:
    def __init__(
        self,
        name="",
        estimator="",
        inherit=False,
        metrics=[],
        hyperparameters={},
        datasets=[],
        random_state=None,
        profiling_file_type="",
        profiling_output_extensions=[],
    ):
        self.name = name
        self.estimator = estimator
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets
        self.random_state = random_state
        self.profiling_file_type = profiling_file_type
        self.profiling_output_extensions = profiling_output_extensions

    def _make_params_grid(self):
        params = self.hyperparameters.get("init", {})
        if not params:
            estimator_class = self._load_estimator_class()
            estimator = estimator_class()
            # Parameters grid should have list values
            params = {k: [v] for k, v in estimator.__dict__.items()}
        grid = list(ParameterGrid(params))
        np.random.shuffle(grid)
        return grid

    def _set_lib(self):
        self.lib_ = self.estimator.split(".")[0]

    def _load_estimator_class(self):
        split_path = self.estimator.split(".")
        mod, class_name = ".".join(split_path[:-1]), split_path[-1]
        return getattr(importlib.import_module(mod), class_name)

    def _load_metrics_funcs(self):
        module = importlib.import_module("sklearn.metrics")
        return [getattr(module, m) for m in self.metrics]

    def run(self):
        self._set_lib()
        estimator_class = self._load_estimator_class()
        metrics_funcs = self._load_metrics_funcs()
        params_grid = self._make_params_grid()
        self.results_ = []
        start = time.perf_counter()
        for dataset in self.datasets:
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = list(reversed(sorted(dataset["n_samples_test"])))
            n_samples_valid = dataset.get("n_samples_valid", None)
            is_hpo_curve = dataset.get("hpo_curve", False)
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
                if n_samples_valid is not None:
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X_train,
                        y_train,
                        test_size=n_samples_valid,
                        random_state=self.random_state,
                    )
                fit_params = {}
                for k, v in self.hyperparameters.get("fit", {}).items():
                    fit_params[k] = eval(str(v))

                for params in params_grid:
                    estimator = estimator_class(**params)
                    set_random_state(estimator, random_state=self.random_state)
                    bench_func = estimator.fit
                    # Use digests to identify results later in reporting
                    hyperparams_digest = joblib.hash(params)
                    dataset_digest = joblib.hash(dataset)
                    profiling_output_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_fit_{hyperparams_digest}_{dataset_digest}"

                    benchmark_info = BenchFuncExecutor().run(
                        bench_func,
                        estimator,
                        profiling_output_path,
                        self.profiling_output_extensions,
                        X_train,
                        y=y_train,
                        max_iter=1,
                        **fit_params,
                    )

                    row = dict(
                        estimator=self.name,
                        function=bench_func.__name__,
                        n_samples_train=ns_train,
                        n_samples=ns_train,
                        n_features=n_features,
                        hyperparams_digest=hyperparams_digest,
                        dataset_digest=dataset_digest,
                        **benchmark_info,
                        **params,
                    )

                    self.results_.append(row)

                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = predict_or_transform(estimator)

                        profiling_output_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_{bench_func.__name__}_{hyperparams_digest}_{dataset_digest}"
                        executor = BenchFuncExecutor()
                        bench_func_params = (
                            self.hyperparameters[bench_func.__name__]
                            if bench_func.__name__ in self.hyperparameters
                            else {}
                        )
                        benchmark_info = executor.run(
                            bench_func,
                            estimator,
                            profiling_output_path,
                            self.profiling_output_extensions,
                            X_test_,
                            max_iter=1 if is_hpo_curve else BENCHMARK_MAX_ITER,
                            **bench_func_params,
                        )
                        row = dict(
                            estimator=self.name,
                            function=bench_func.__name__,
                            n_samples_train=ns_train,
                            n_samples=ns_test,
                            n_features=n_features,
                            hyperparams_digest=hyperparams_digest,
                            dataset_digest=dataset_digest,
                            **benchmark_info,
                            **params,
                        )

                        for metric_func in metrics_funcs:
                            y_pred = executor.func_result_
                            score = metric_func(y_test_, y_pred)
                            row[metric_func.__name__] = score

                        pprint(row)
                        self.results_.append(row)
                        self.to_csv()

                        if is_hpo_curve:
                            now = time.perf_counter()
                            if now - start > BENCHMARK_TIME_BUDGET:
                                return
        return self

    def to_csv(self):
        csv_path = f"{BENCHMARKING_RESULTS_PATH}/{self.lib_}_{self.name}.csv"
        pd.DataFrame(self.results_).to_csv(
            csv_path,
            mode="w+",
            index=False,
        )
