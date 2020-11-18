import numpy as np
import matplotlib.pyplot as plt


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
            alldfs[i][j] = alldfs[i][j][["n_samples", "n_features", "speedup"]].groupby(["n_samples", "n_features"]).last()

    fig, ax = plt.subplots(n_estimators, n_vals, figsize=(8*n_vals, 6*n_estimators))
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
