import numpy as np
import os
import glob
from pathlib import Path
import glob
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import importlib
import itertools
import re
from joblib import Memory


class ExecutionTimer:
    @staticmethod
    def run(func, *args):
        times = []
        start = time.perf_counter()
        for _ in range(10):
            start_ = time.perf_counter()
            result = func(*args)
            end_ = time.perf_counter()
            times.append(end_ - start_)
            curr = time.perf_counter()
            if curr - start > 3:
                break
        mean_time, std_time = np.mean(times), np.std(times)
        return (result, mean_time, std_time)


_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


def get_config_path():
    path = Path(__file__).resolve().parent
    file = os.environ["CONFIG_FILE"]
    path = path / file
    return path


@memory.cache
def gen_data(generator_path, n_samples=1000, n_features=10, **kwargs):
    """Returns a tuple of data from the specified generator."""
    splitted_path = generator_path.split(".")
    module, func = ".".join(splitted_path[:-1]), splitted_path[-1]
    generator_func = getattr(importlib.import_module(module), func)
    data = generator_func(n_samples=n_samples, n_features=n_features, **kwargs)
    return data


def is_scientific_notation(string):
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def predict_or_transform(estimator):
    if hasattr(estimator, "predict"):
        bench_func = estimator.predict
    else:
        bench_func = estimator.transform
    return bench_func


def print_time_report():
    path = Path(__file__).resolve().parent
    path = path / "results/time_report.csv"
    df = pd.read_csv(str(path), index_col="algo")
    return df


def _gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def _make_suffix(lib):
    return "_" + lib


def _make_dataset(
    algo,
    lib,
    speedup_col="mean_time_elapsed",
    speedup_err_col="std_time_elapsed",
    def_merge_cols=["estimator", "function", "n_samples", "n_features"],
    add_merge_cols=[],
):
    merge_cols = def_merge_cols + add_merge_cols
    results_path = Path(__file__).resolve().parent / "results"
    lib_df = pd.read_csv("%s/%s/%s.csv" % (str(results_path), lib, algo))
    skl_df = pd.read_csv("%s/sklearn/%s.csv" % (str(results_path), algo))
    lib_suffix = _make_suffix(lib)
    merged_df = skl_df.merge(lib_df, on=merge_cols, suffixes=["_sklearn", lib_suffix])

    skl_col = speedup_col + "_sklearn"
    lib_col = speedup_col + lib_suffix
    merged_df["speedup"] = merged_df[skl_col] / merged_df[lib_col]

    skl_col = speedup_err_col + "_sklearn"
    lib_col = speedup_err_col + lib_suffix
    merged_df["speedup_err"] = merged_df[skl_col] / merged_df[lib_col]

    return merged_df


def plot_knn(algo="KNeighborsClassifier", n_cols=2):
    merged_df_knn = _make_dataset(
        algo,
        "daal4py",
        add_merge_cols=[
            "algorithm",
            "n_jobs",
            "n_neighbors",
        ],
    )
    merged_df_knn = merged_df_knn[
        [
            "function",
            "n_samples",
            "n_features",
            "algorithm",
            "n_jobs",
            "n_neighbors",
            "mean_time_elapsed_sklearn",
            "mean_time_elapsed_daal4py",
            "speedup",
            "speedup_err",
        ]
    ]

    merged_df_knn_grouped = merged_df_knn.groupby(
        ["algorithm", "n_neighbors", "function"]
    )

    n_plots = len(merged_df_knn_grouped)
    n_rows = n_plots // n_cols + n_plots % n_cols
    coordinates = _gen_coordinates_grid(n_rows, n_cols)

    subplot_titles = [
        "algo: %s, k: %s, func: %s" % params for params, _ in merged_df_knn_grouped
    ]
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, y_title="Speedup"
    )

    for (row, col), (_, df) in zip(coordinates, merged_df_knn_grouped):
        df["speedup"] = df["speedup"].round(2)
        df["speedup_err"] = df["speedup_err"].round(2)
        df[["mean_time_elapsed_sklearn", "mean_time_elapsed_daal4py"]] = df[
            ["mean_time_elapsed_sklearn", "mean_time_elapsed_daal4py"]
        ].round(4)

        x1 = df[["n_samples", "n_features"]][df["n_jobs"] == 1]
        x1 = [f"({ns}, {nf})" for ns, nf in x1.values]
        y1 = df["speedup"][df["n_jobs"] == 1]
        fig.add_trace(
            go.Bar(
                x=x1,
                y=y1,
                error_y=dict(type="data", array=df[df["n_jobs"] == 1]["speedup_err"]),
                name="n_jobs: 1",
                marker_color="indianred",
                hovertemplate="function: <b>%{customdata[0]}</b><br>"
                + "n_samples: <b>%{customdata[1]}</b><br>"
                + "n_features: <b>%{customdata[2]}</b><br>"
                + "algo: <b>%{customdata[3]}</b><br>"
                + "n_jobs: <b>%{customdata[4]}</b><br>"
                + "n_neighbors: <b>%{customdata[5]}</b><br>"
                + "Time sklearn (s): <b>%{customdata[6]}</b><br>"
                + "Time d4p (s): <b>%{customdata[7]}</b><br>"
                + "speedup: <b>%{customdata[8]}</b><br>"
                + "speedup error: <b>%{customdata[9]}</b><br>"
                + "<extra></extra>",
                customdata=df[df["n_jobs"] == 1].values,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        x2 = df[["n_samples", "n_features"]][df["n_jobs"] == 1]
        x2 = [f"({ns}, {nf})" for ns, nf in x2.values]
        y2 = df["speedup"][df["n_jobs"] == -1]
        fig.add_trace(
            go.Bar(
                x=x2,
                y=y2,
                error_y=dict(type="data", array=df[df["n_jobs"] == -1]["speedup_err"]),
                name="n_jobs: -1",
                marker_color="lightsalmon",
                hovertemplate="function: <b>%{customdata[0]}</b><br>"
                + "n_samples: <b>%{customdata[1]}</b><br>"
                + "n_features: <b>%{customdata[2]}</b><br>"
                + "algo: <b>%{customdata[3]}</b><br>"
                + "n_jobs: <b>%{customdata[4]}</b><br>"
                + "n_neighbors: <b>%{customdata[5]}</b><br>"
                + "Time sklearn (s): <b>%{customdata[6]}</b><br>"
                + "Time d4p (s): <b>%{customdata[7]}</b><br>"
                + "speedup: <b>%{customdata[9]}</b><br>"
                + "speedup error: <b>%{customdata[10]}</b><br>"
                + "<extra></extra>",
                customdata=df[df["n_jobs"] == -1].values,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(height=1500, barmode="group", showlegend=False)
    fig.show()


def plot_kmeans():
    merged_df_kmeans = _make_dataset(
        "kmeans",
        "daal4py",
        add_merge_cols=[
            "init",
            "max_iter",
            "n_clusters",
            "n_init",
            "tol",
        ],
    )
    merged_df_kmeans = merged_df_kmeans[
        [
            "function",
            "n_samples",
            "n_features",
            "init",
            "max_iter",
            "n_clusters",
            "n_init",
            "tol",
            "mean_time_elapsed_skl",
            "mean_time_elapsed_d4p",
            "speedup",
        ]
    ]
    merged_df_kmeans_grouped = merged_df_kmeans.groupby(
        ["init", "max_iter", "n_clusters", "n_init", "tol", "function"]
    )

    coordinates = _gen_coordinates_grid(6, 2)

    subplot_titles = []
    for params, _ in merged_df_kmeans_grouped:
        subplot_titles.append("init: %s - function: %s" % (params[0], params[-1]))

    fig = make_subplots(
        rows=6,
        cols=2,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        y_title="Speedup",
    )

    for (row, col), (_, df) in zip(coordinates, merged_df_kmeans_grouped):
        df["speedup"] = df["speedup"].round(2)
        df[["mean_time_elapsed_skl", "mean_time_elapsed_d4p"]] = df[
            ["mean_time_elapsed_skl", "mean_time_elapsed_d4p"]
        ].round(6)
        x = df[["n_samples", "n_features"]]
        x = [f"({ns}, {nf})" for ns, nf in x.values]
        y = df["speedup"]
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                hovertemplate="function: <b>%{customdata[0]}</b><br>"
                + "n_samples: <b>%{customdata[1]}</b><br>"
                + "n_features: <b>%{customdata[2]}</b><br>"
                + "init: <b>%{customdata[3]}</b><br>"
                + "max_iter: <b>%{customdata[4]}</b><br>"
                + "n_clusters: <b>%{customdata[5]}</b><br>"
                + "n_init: <b>%{customdata[6]}</b><br>"
                + "tol: <b>%{customdata[7]}</b><br>"
                + "Time sklearn (s): <b>%{customdata[8]}</b><br>"
                + "Time d4p (s): <b>%{customdata[9]}</b><br>"
                + "speedup: <b>%{customdata[10]}</b><br>"
                + "<extra></extra>",
                customdata=df.values,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(height=2000, barmode="group", showlegend=False)
    fig.show()


def clean_results():
    current_path = Path(__file__).resolve().parent
    files_path = current_path / "results/**/*.csv"
    files = glob.glob(str(files_path), recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return hour, min, sec