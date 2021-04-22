import os
import yaml
import glob
import glob
import importlib
import re
from joblib import Memory
from sklearn_benchmarks.config import RESULTS_PATH

_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


@memory.cache
def gen_data(
    generator_path, n_samples=1000, n_features=10, random_state=None, **kwargs
):
    """Returns a tuple of data from the specified generator."""
    splitted_path = generator_path.split(".")
    module, func = ".".join(splitted_path[:-1]), splitted_path[-1]
    generator_func = getattr(importlib.import_module(module), func)
    data = generator_func(
        n_samples=n_samples, n_features=n_features, random_state=random_state, **kwargs
    )
    return data


def is_scientific_notation(string):
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def predict_or_transform(estimator):
    if hasattr(estimator, "predict"):
        bench_func = estimator.predict
    else:
        bench_func = estimator.transform
    return bench_func


def clean_results():
    extensions = [".csv", ".html", ".json.gz"]
    files = []
    for extension in extensions:
        files_path = str(RESULTS_PATH / "**/*") + extension
        files += glob.glob(str(files_path), recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return hour, min, sec
