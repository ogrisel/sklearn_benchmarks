from pathlib import Path
import yaml
import time
import importlib
import time
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn_benchmarks.utils import gen_data, is_scientific_notation
from sklearn.model_selection import train_test_split


class Timer:
    @staticmethod
    def run(func, *args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        return (result, end - start)


class Benchmark:
    """Runs benchmarks on one estimator for one library, accross potentially multiple datasets"""

    def __init__(
        self,
        name,
        source_path,
        inherit=False,
        metrics=["accuracy"],
        hyperparameters={},
        datasets=[],
    ):
        self.name = name
        self.source_path = source_path
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets

    def _lib_name(self):
        return self.source_path.split(".")[0]

    def _load_estimator_class(self):
        splitted_path = self.source_path.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        return getattr(importlib.import_module(module), class_name)

    def _init_parameters_grid(self):
        return ParameterGrid(self.hyperparameters)

    def _validate_params(self):
        if not isinstance(self.name, str):
            raise ValueError("name should be a string")
        if not self.name:
            raise ValueError("name should not be an empty string")
        if not isinstance(self.source_path, str):
            raise ValueError("source_path should be a string")
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

    def _predict_or_transform(self, estimator):
        if hasattr(estimator, "predict"):
            bench_func = estimator.predict
        else:
            bench_func = estimator.transform
        return bench_func

    def run(self):
        self._validate_params()
        estimator_class = self._load_estimator_class()
        metrics_functions = self._load_metrics_functions()
        parameters_grid = self._init_parameters_grid()
        self.results_ = []
        for dataset in self.datasets:
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = reversed(sorted(dataset["n_samples_test"]))
            for ns_train in n_samples_train:
                X, y = gen_data(
                    dataset["generator"],
                    n_samples=ns_train + max(dataset["n_samples_test"]),
                    n_features=n_features,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=ns_train
                )
                for params in parameters_grid:
                    estimator = estimator_class(**params)
                    _, time_elapsed = Timer.run(estimator.fit, X_train, y_train)
                    row = dict(
                        estimator=self.name,
                        lib=self._lib_name(),
                        function="fit",
                        time_elapsed=time_elapsed,
                        n_samples=ns_train,
                        n_features=n_features,
                        **params,
                    )
                    print(row)
                    print("---")
                    self.results_.append(row)
                    for i, ns_test in enumerate(n_samples_test):
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = self._predict_or_transform(estimator)
                        y_pred, time_elapsed = Timer.run(bench_func, X_test_)
                        if i == 0:
                            scores = {
                                func.__name__: func(y_test_, y_pred)
                                for func in metrics_functions
                            }
                        row = dict(
                            estimator=self.name,
                            lib=self._lib_name(),
                            function="predict",
                            time_elapsed=time_elapsed,
                            n_samples=ns_test,
                            n_features=n_features,
                            **scores,
                            **params,
                        )
                        print(row)
                        print("---")
                        self.results_.append(row)
        return self

    def to_csv(self):
        results = pd.DataFrame(self.results_)
        results.to_csv(
            f"sklearn_benchmarks/results/{self._lib_name()}/{self.name}.csv",
            mode="w+",
            index=False,
        )


def _prepare_params(params):
    for key, value in params["hyperparameters"].items():
        if not isinstance(value, list):
            continue
        for i, el in enumerate(value):
            if is_scientific_notation(el):
                if "-" in el:
                    params["hyperparameters"][key][i] = float(el)
                else:
                    params["hyperparameters"][key][i] = int(float(el))

    for dataset in params["datasets"]:
        dataset["n_features"] = int(float(dataset["n_features"]))
        for i, ns_train in enumerate(dataset["n_samples_train"]):
            dataset["n_samples_train"][i] = int(float(ns_train))
        for i, ns_test in enumerate(dataset["n_samples_test"]):
            dataset["n_samples_test"][i] = int(float(ns_test))

    return params


def main():
    current_path = Path(__file__).resolve().parent
    config_path = current_path / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.full_load(config_file)

    estimators = config["estimators"]

    for params in estimators.values():
        if "inherit" in params:
            name_inherits_from = params["inherit"]
            # Merge params with estimator from which inherits
            params = estimators[name_inherits_from] | params

        params = _prepare_params(params)

        benchmark_estimator = Benchmark(**params)
        benchmark_estimator.run()
        benchmark_estimator.to_csv()


if __name__ == "__main__":
    main()
