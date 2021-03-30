import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_blobs, make_classification
from joblib import Memory


_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


def _gen_data_regression(n_samples=1000, n_features=10, random_state=42):
    """Wrapper for sklearn make_regression"""
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=random_state
    )
    return X, y


def _gen_data_blobs(n_samples=1000, n_features=10, random_state=42, centers=None):
    """Wrapper for sklearn make_blobs"""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state,
    )
    return X, y


def _gen_data_classification(
    n_samples=1000, n_features=10, random_state=42, n_classes=2
):
    """Wrapper for sklearn make_blobs"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
    )
    return X, y


_data_generators = {
    "blobs": _gen_data_blobs,
    "classification": _gen_data_classification,
    "regression": _gen_data_regression,
}


@memory.cache
def gen_data(dataset_name, n_samples=1000, n_features=10, random_state=42, **kwargs):
    """Returns a tuple of data from the specified generator."""
    n_samples, n_features = int(float(n_samples)), int(float(n_features))
    data = _data_generators[dataset_name](n_samples, n_features, random_state, **kwargs)

    return data


def is_scientific_notation(string):
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def predict_or_transform(estimator):
    if hasattr(estimator, "predict"):
        bench_func = estimator.predict
    else:
        bench_func = estimator.transform
    return bench_func