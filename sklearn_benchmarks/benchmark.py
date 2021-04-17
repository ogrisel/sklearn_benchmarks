import numpy as np
import pandas as pd
import joblib
import importlib
import time
from viztracer import VizTracer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import set_random_state
from sklearn_benchmarks.config import BENCHMARKING_RESULTS_PATH, PROFILING_RESULTS_PATH
from sklearn_benchmarks.utils.misc import gen_data, predict_or_transform


class FuncExecutor:
    @staticmethod
    def run(func, profiling_output_file, *args):
        # First run with a profiler (not timed)
        with VizTracer(output_file=profiling_output_file, verbose=0) as tracer:
            tracer.start()
            result = func(*args)
            tracer.stop()
            tracer.save()

        # Next runs: at most 10 runs or 30 sec
        times = []
        start = time.perf_counter()
        for _ in range(10):
            start_ = time.perf_counter()
            result = func(*args)
            end_ = time.perf_counter()
            times.append(end_ - start_)
            if end_ - start > 30:
                break
        mean_time, stdev_time = np.mean(times), np.std(times)
        return (result, mean_time, stdev_time)


class Benchmark:
    """
    Runs benchmarks on one estimator for one library, accross potentially multiple datasets
    """

    def __init__(
        self,
        name,
        estimator,
        inherit=False,
        metrics=["accuracy"],
        hyperparameters={},
        datasets=[],
    ):
        self.name = name
        self.estimator = estimator
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets

    def _set_lib(self):
        self.lib_ = self.estimator.split(".")[0]

    def _load_estimator_class(self):
        splitted_path = self.estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        return getattr(importlib.import_module(module), class_name)

    def _init_parameters_grid(self):
        return ParameterGrid(self.hyperparameters)

    def _validate_params(self):
        if not isinstance(self.name, str):
            raise ValueError("name should be a string")
        if not self.name:
            raise ValueError("name should not be an empty string")
        if not isinstance(self.estimator, str):
            raise ValueError("estimator should be a string")
        if not (isinstance(self.inherit, bool) or isinstance(self.inherit, str)):
            raise ValueError("inherit should be a either True, False or a string")
        if not isinstance(self.metrics, list):
            raise ValueError("metrics should be a list")
        if not isinstance(self.datasets, list):
            raise ValueError("datasets should be a list")
        if not isinstance(self.hyperparameters, object):
            raise ValueError("hyperparameters should be an object")

    def _load_metrics_functions(self):
        module = importlib.import_module("sklearn.metrics")
        return [getattr(module, m) for m in self.metrics]

    def run(self):
        self._validate_params()
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
                    X, y, train_size=ns_train
                )
                for params in parameters_grid:
                    estimator = estimator_class(**params)
                    set_random_state(estimator, random_state=42)
                    hyperparams_digest = joblib.hash(params)
                    dataset_digest = joblib.hash([ns_train, n_features])
                    profiling_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_fit_{hyperparams_digest}_{dataset_digest}.json.gz"

                    _, mean, stdev = FuncExecutor.run(
                        estimator.fit, profiling_path, X_train, y_train
                    )
                    row = dict(
                        estimator=self.name,
                        lib=self.lib_,
                        function="fit",
                        mean=mean,
                        stdev=stdev,
                        n_samples=ns_train,
                        n_features=n_features,
                        hyperparams_digest=hyperparams_digest,
                        dataset_digest=dataset_digest,
                        **params,
                    )
                    if hasattr(estimator, "n_iter_"):
                        row["n_iter"] = estimator.n_iter_

                    self.results_.append(row)

                    print(
                        "%s - %s - %s - mean: %6.3f - stdev: %6.3f"
                        % (self.lib_, self.name, "fit", mean, stdev)
                    )

                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = predict_or_transform(estimator)
                        dataset_digest = joblib.hash([ns_test, n_features])
                        profiling_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_{bench_func.__name__}_{hyperparams_digest}_{dataset_digest}.json.gz"

                        (
                            y_pred,
                            mean,
                            stdev,
                        ) = FuncExecutor.run(bench_func, profiling_path, X_test_)
                        if i == 0:
                            scores = {
                                func.__name__: func(y_test_, y_pred)
                                for func in metrics_functions
                            }
                        row = dict(
                            estimator=self.name,
                            lib=self.lib_,
                            function="predict",
                            mean=mean,
                            stdev=stdev,
                            n_samples=ns_test,
                            n_features=n_features,
                            hyperparams_digest=hyperparams_digest,
                            dataset_digest=dataset_digest,
                            **scores,
                            **params,
                        )
                        print(
                            "%s - %s - %s - mean: %6.3f - stdev: %6.3f"
                            % (self.lib_, self.name, bench_func.__name__, mean, stdev)
                        )
                        self.results_.append(row)
        return self

    def to_csv(self):
        csv_path = f"{BENCHMARKING_RESULTS_PATH}/{self.lib_}_{self.name}.csv"
        pd.DataFrame(self.results_).to_csv(
            csv_path,
            mode="w+",
            index=False,
        )