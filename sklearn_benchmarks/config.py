from pathlib import Path
from sklearn_benchmarks.utils.misc import is_scientific_notation

RESULTS_PATH = Path(__file__).resolve().parent / "results"
PROFILING_RESULTS_PATH = RESULTS_PATH / "profiling"
BENCHMARKING_RESULTS_PATH = RESULTS_PATH / "benchmarking"
TIME_REPORT_PATH = RESULTS_PATH / "time_report.csv"
DEFAULT_CONFIG_FILE_PATH = "config.yml"
BASE_LIB = "sklearn"


def prepare_params(params):
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