from pathlib import Path
import yaml
import time
import importlib
import re
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn_benchmarks.utils import gen_data
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


class BenchmarkEstimator:
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
        components = self.source_path.split(".")
        module = ".".join(components[:-1])
        class_name = components[-1]
        try:
            self.estimator_class_ = getattr(importlib.import_module(module), class_name)
        except Exception:
            raise ValueError("Source class provided in not valid")

    def _init_params_grid(self):
        self.params_grid_ = ParameterGrid(self.hyperparameters)

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

    def run(self):
        self._validate_params()
        self._load_estimator_class()
        self._init_params_grid()
        self.results_ = []
        # for each dataset
        for dataset in self.datasets:
            print("start dataset: ", dataset["generator"])
            # for each n_samples in dataset
            for ns_train in dataset["n_samples_train"]:
                print("start ns_train: ", ns_train)
                ns_train = int(float(ns_train))
                max_n_samples_test = max(
                    map(lambda ns_test: int(float(ns_test)), dataset["n_samples_test"])
                )
                n_features = int(float(dataset["n_features"]))
                # generate a dataset with generator:
                # X, y = make_classification(n_samples = n_samples_train + max(n_samples_test) ...)
                X, y = gen_data(
                    dataset["generator"],
                    n_samples=ns_train + max_n_samples_test,
                    n_features=n_features,
                )
                # X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=n_samples_train, random_state=0)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=ns_train
                )
                # fit the estimator with train data (record time)
                for params in self.params_grid_:
                    print("start params: ", params)
                    estimator = self.estimator_class_(**params)
                    t0 = time.perf_counter()
                    estimator.fit(X_train, y_train)
                    t1 = time.perf_counter()
                    row = dict(
                        estimator=self.name,
                        lib=self._lib_name(),
                        function="fit",
                        time_elapsed=t1 - t0,
                        n_samples=ns_train,
                        n_features=n_features,
                        **params,
                    )
                    self.results_.append(row)
                    # for each n_samples_test
                    test_results = []
                    for ns_test in sorted(
                        map(
                            lambda ns_test: int(float(ns_test)),
                            dataset["n_samples_test"],
                        )
                    ):
                        print("start ns_test: ", ns_test)
                        # slice X_test and y_test
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        # predict test data (record time)
                        t0 = time.perf_counter()
                        bench_func = (
                            estimator.predict
                            if hasattr(estimator, "predict")
                            else estimator.transform
                        )
                        y_pred = bench_func(X_test_)
                        t1 = time.perf_counter()
                        print("end ns_test: ", ns_test)
                        # load metrics function from sklearn.metrics
                        # for biggest test set, compute metrics
                        row = dict(
                            estimator=self.name,
                            lib=self._lib_name(),
                            function="predict",
                            time_elapsed=t1 - t0,
                            n_samples=ns_test,
                            n_features=n_features,
                            **params,
                        )
                        test_results.append(row)
                    for metric in self.metrics:
                        print("start metric: ", metric)
                        metric_func = getattr(
                            importlib.import_module("sklearn.metrics"), metric
                        )
                        score = metric_func(y_test_, y_pred)
                        for el in test_results:
                            el[metric] = score
                        print("end metric: ", metric)
                    for el in test_results:
                        self.results_.append(el)
                    print("end params: ", params)
                print("end ns_train: ", ns_train)
            print("end dataset: ", dataset["generator"])
        return self

    def to_csv(self):
        pd.DataFrame(self.results_).to_csv(
            f"sklearn_benchmarks/results/{self._lib_name()}/{self.name}.csv",
            mode="w+",
            index=False,
        )


def main():
    current_path = Path(__file__).resolve().parent
    config_path = current_path / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.full_load(config_file)

    estimators = config["estimators"]

    for name, params in estimators.items():
        if "hyperparameters" in params:
            hyperparameters = params["hyperparameters"]
            for key, value in hyperparameters.items():
                if not isinstance(value, list):
                    continue
                for i, el in enumerate(value):
                    if isinstance(el, str) and bool(re.match(r"1[eE](\-)*\d{1,}", el)):
                        if "-" in el:
                            hyperparameters[key][i] = float(el)
                        else:
                            hyperparameters[key][i] = int(float(el))
            params["hyperparameters"] = hyperparameters
        print(name)
        if "inherit" in params:
            params = estimators[params["inherit"]] | params
        benchmark_estimator = BenchmarkEstimator(**params)
        benchmark_estimator.run()
        benchmark_estimator.to_csv()
        print("--------------------------")


if __name__ == "__main__":
    main()
