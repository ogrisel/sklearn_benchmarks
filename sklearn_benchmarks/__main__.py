from pathlib import Path
import yaml
import json
import time
import importlib
import time
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn_benchmarks.utils import (
    FuncExecutor,
    gen_data,
    is_scientific_notation,
    predict_or_transform,
    clean_results,
    convert,
    get_config_path,
)
from sklearn.model_selection import train_test_split


class Benchmark:
    """Runs benchmarks on one estimator for one library, accross potentially multiple datasets"""

    RESULTS_PATH = Path(__file__).resolve().parent / "results"

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

    def _lib(self):
        return self.estimator.split(".")[0]

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

                    profiling_path = self.RESULTS_PATH / "profiling"
                    Path(profiling_path).mkdir(parents=True, exist_ok=True)
                    prof_out_path = f"{str(profiling_path)}/fit_{self.name}.html"

                    _, mean_time, stdev_time = FuncExecutor.run(
                        estimator.fit, prof_out_path, X_train, y_train
                    )
                    row = dict(
                        estimator=self.name,
                        lib=self._lib(),
                        function="fit",
                        mean_time=mean_time,
                        stdev_time=stdev_time,
                        n_samples=ns_train,
                        n_features=n_features,
                        **params,
                    )
                    if hasattr(estimator, "n_iter_"):
                        row["n_iter"] = estimator.n_iter_
                    print(json.dumps(row, indent=4, ensure_ascii=False))
                    print("---")
                    self.results_.append(row)
                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = predict_or_transform(estimator)

                        prof_out_path = f"{str(profiling_path)}/{bench_func.__name__}_{self.name}.html"

                        (
                            y_pred,
                            mean_time,
                            stdev_time,
                        ) = FuncExecutor.run(bench_func, prof_out_path, X_test_)
                        if i == 0:
                            scores = {
                                func.__name__: func(y_test_, y_pred)
                                for func in metrics_functions
                            }
                        row = dict(
                            estimator=self.name,
                            lib=self._lib(),
                            function="predict",
                            mean_time=mean_time,
                            stdev_time=stdev_time,
                            n_samples=ns_test,
                            n_features=n_features,
                            **scores,
                            **params,
                        )
                        print(json.dumps(row, indent=4, ensure_ascii=False))
                        print("---")
                        self.results_.append(row)
        return self

    def to_csv(self):
        csv_path = f"{self.RESULTS_PATH}/{self._lib()}_{self.name}.csv"
        pd.DataFrame(self.results_).to_csv(
            csv_path,
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
    clean_results()

    config_path = get_config_path()
    with open(config_path, "r") as config_file:
        config = yaml.full_load(config_file)

    estimators = config["estimators"]

    time_report = pd.DataFrame(columns=["algo", "hour", "min", "sec"])
    t0 = time.perf_counter()
    for name, params in estimators.items():
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = estimators[params["inherit"]]
            params["estimator"] = curr_estimator

        params = _prepare_params(params)

        benchmark_estimator = Benchmark(**params)
        print(f"start benchmark {name}")
        print("---")
        t0_ = time.perf_counter()
        benchmark_estimator.run()
        t1_ = time.perf_counter()
        time_report.loc[len(time_report)] = [name, *convert(t1_ - t0_)]
        benchmark_estimator.to_csv()

    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", *convert(t1 - t0)]
    time_report = time_report.round(2)
    current_path = Path(__file__).resolve().parent
    time_report_path = current_path / "results/time_report.csv"
    time_report.to_csv(
        str(time_report_path),
        mode="w+",
        index=False,
    )


if __name__ == "__main__":
    main()
