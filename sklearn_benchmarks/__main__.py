import yaml
import importlib
from sklearn.model_selection import ParameterGrid
from sklearn_benchmarks.utils import gen_data
from sklearn.model_selection import train_test_split

CONFIG_FILE_PATH = "sklearn_benchmarks/config.yaml"


class BenchmarkEstimator:
    def __init__(
        self,
        name,
        source_path,
        inherit=False,
        metrics="accuracy",
        accepts_labels=True,
        hyperparameters={},
        datasets=[],
    ):
        self.name = name
        self.source_path = source_path
        self.inherit = inherit
        self.metrics = metrics
        self.accepts_labels = accepts_labels
        self.hyperparameters = hyperparameters
        self.datasets = datasets

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
        if not (isinstance(self.metrics, str) or isinstance(self.metrics, list)):
            raise ValueError("metrics should be a string or a list")
        if not isinstance(self.accepts_labels, bool):
            raise ValueError("accepts_labels should be a either True or False")
        if not isinstance(self.datasets, list):
            raise ValueError("datasets should be a list")
        if not isinstance(self.hyperparameters, object):
            raise ValueError("hyperparameters should be an object")

    def run(self):
        self._validate_params()
        self._load_estimator_class()
        self._init_params_grid()
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
                    estimator.fit(X_train, y_train)
                    # for each n_samples_test
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
                        y_pred = estimator.predict(X_test_)
                        print("end ns_test: ", ns_test)
                    # load metrics function from sklearn.metrics
                    # for biggest test set, compute metrics
                    for metric in dataset["metrics"]:
                        print("start metric: ", metric)
                        metric_func = getattr(
                            importlib.import_module("sklearn.metrics"), metric
                        )
                        metric_func(y_test_, y_pred)
                        print("end metric: ", metric)
                    print("end params: ", params)
                print("end ns_train: ", ns_train)
            print("end dataset: ", dataset["generator"])
        return self


def main():
    with open(CONFIG_FILE_PATH, "r") as file:
        config = yaml.full_load(file)

    benchmark_estimators = [
        BenchmarkEstimator(**params) for params in config["estimators"]
    ]
    for benchmark_estimator in benchmark_estimators:
        benchmark_estimator.run()


if __name__ == "__main__":
    main()
