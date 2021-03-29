import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_blobs, make_classification
from joblib import Memory

### PLOTTING ###


def plot(benchmark_result, split=None):
    df = benchmark_result

    estimators = df["algo"].unique()
    n_estimators = len(estimators)
    alldfs = [df[df["algo"] == val] for val in estimators]

    if split is not None:
        vals = df[split].unique()
        n_vals = len(vals)
        for i, estdf in enumerate(alldfs):
            alldfs[i] = [estdf[estdf[split] == val] for val in vals]
    else:
        n_vals = 1
        alldfs = [[estdf] for estdf in alldfs]

    for i, estdfs in enumerate(alldfs):
        for j in range(len(estdfs)):
            alldfs[i][j] = (
                alldfs[i][j][["n_samples", "n_features", "speedup"]]
                .groupby(["n_samples", "n_features"])
                .last()
            )

    fig, ax = plt.subplots(n_estimators, n_vals, figsize=(8 * n_vals, 6 * n_estimators))
    if n_estimators == 1 and n_vals == 1:
        ax = np.array([[ax]])
    elif n_estimators == 1:
        ax = np.array([ax])
    elif n_vals == 1:
        ax = np.array([ax]).reshape(-1, 1)

    for i, estdfs in enumerate(alldfs):
        for j, df in enumerate(estdfs):
            df.plot.bar(ax=ax[i, j])
            if split is not None:
                ax[i, j].set_title(f"{estimators[i]} | {split}={vals[j]}")
            else:
                ax[i, j].set_title(f"{estimators[i]}")

    fig.tight_layout()
    plt.show()


### DATA GENERATION ###

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
