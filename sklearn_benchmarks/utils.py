import yaml
import os
import glob
import glob
import time
import importlib
import itertools
import re
import qgrid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
from pathlib import Path
from plotly.subplots import make_subplots
from joblib import Memory
from viztracer import VizTracer

RESULTS_PATH = Path(__file__).resolve().parent / "results"


class FuncExecutor:
    @staticmethod
    def run(func, profiling_output_file, *args):
        # First run with a profiler (not timed)
        with VizTracer(output_file=profiling_output_file, verbose=0) as tracer:
            tracer.start()
            result = func(*args)
            tracer.stop()
            tracer.save()

        # Next runs: at most 10 runs or 30 sec
        times = []
        start = time.perf_counter()
        for _ in range(10):
            start_ = time.perf_counter()
            result = func(*args)
            end_ = time.perf_counter()
            times.append(end_ - start_)
            if end_ - start > 30:
                break
        mean_time, stdev_time = np.mean(times), np.std(times)
        return (result, mean_time, stdev_time)


_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


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


def print_results(algo="", versus_lib="", compare_cols=[]):
    data = _make_dataset(algo, versus_lib, compare_cols=compare_cols)
    qgrid_widget = qgrid.show_grid(data, show_toolbar=True)
    display(qgrid_widget)


def _gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def _make_dataset(
    algo, lib, speedup_col="mean", stdev_speedup_col="stdev", compare_cols=[]
):
    lib_df = pd.read_csv("%s/%s_%s.csv" % (str(RESULTS_PATH), lib, algo))
    skl_df = pd.read_csv("%s/sklearn_%s.csv" % (str(RESULTS_PATH), algo))
    lib_suffix = "_" + lib
    merge_cols = [
        speedup_col,
        stdev_speedup_col,
        "hyperparams_digest",
        "dims_digest",
        *compare_cols,
    ]
    merged_df = pd.merge(
        skl_df[merge_cols],
        lib_df[merge_cols],
        on=["hyperparams_digest", "dims_digest"],
        suffixes=["", lib_suffix],
    )
    # merged_df = merged_df.drop(["hyperparams_digest", "dims_digest"], axis=1)
    skl_df = skl_df.drop(merge_cols, axis=1)
    merged_df = pd.merge(skl_df, merged_df, left_index=True, right_index=True)

    numeric_cols = merged_df.select_dtypes(include=["float64"]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].round(4)

    skl_time = merged_df[speedup_col]
    lib_time = merged_df[speedup_col + lib_suffix]
    merged_df["speedup"] = skl_time / lib_time

    skl_std = merged_df[stdev_speedup_col]
    lib_std = merged_df[stdev_speedup_col + lib_suffix]
    merged_df["stdev_speedup"] = (
        merged_df["speedup"] *
        np.sqrt((skl_std / skl_time)**2 + (lib_std / lib_time)**2)
    )

    merged_df["speedup"] = merged_df["speedup"].round(2)
    merged_df["stdev_speedup"] = merged_df["stdev_speedup"].round(2)

    return merged_df


def _make_hover_template(df):
    template = ""
    for index, name in enumerate(df.columns):
        template += "%s: <b>%%{customdata[%i]}</b><br>" % (name, index)
    template += "<extra></extra>"
    return template


def plot_results(
    algo="",
    versus_lib="",
    group_by_cols=[],
    split_hist_by=[],
    compare_cols=[],
    n_cols=2,
):
    merged_df = _make_dataset(algo, versus_lib, compare_cols=compare_cols)
    merged_df_grouped = merged_df.groupby(group_by_cols)

    n_plots = len(merged_df_grouped)
    n_rows = n_plots // n_cols + n_plots % n_cols
    coordinates = _gen_coordinates_grid(n_rows, n_cols)

    subplot_titles = []
    for params, _ in merged_df_grouped:
        title = ""
        for index, (name, val) in enumerate(zip(group_by_cols, params)):
            title += "%s: %s" % (name, val)
            if index > 0 and index % 3 == 0:
                title += "<br>"
            elif index != len(list(zip(group_by_cols, params))) - 1:
                title += " - "

        subplot_titles.append(title)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )

    for (row, col), (_, df) in zip(coordinates, merged_df_grouped):
        df = df.sort_values(by=["n_samples", "n_features"])
        df = df.dropna(axis="columns")
        if split_hist_by:
            for split_col in split_hist_by:
                split_col_vals = df[split_col].unique()
                for index, split_val in enumerate(split_col_vals):
                    x = df[["n_samples", "n_features"]][df[split_col] == split_val]
                    x = [f"({ns}, {nf})" for ns, nf in x.values]
                    y = df["speedup"][df[split_col] == split_val]
                    bar = go.Bar(
                        x=x,
                        y=y,
                        name="%s: %s" % (split_col, split_val),
                        marker_color=px.colors.qualitative.Plotly[index],
                        hovertemplate=_make_hover_template(
                            df[df[split_col] == split_val]
                        ),
                        customdata=df[df[split_col] == split_val].values,
                        showlegend=(row, col) == (1, 1),
                    )
                    fig.add_trace(
                        bar,
                        row=row,
                        col=col,
                    )
        else:
            x = df[["n_samples", "n_features"]]
            x = [f"({ns}, {nf})" for ns, nf in x.values]
            y = df["speedup"]
            bar = go.Bar(
                x=x,
                y=y,
                hovertemplate=_make_hover_template(df),
                customdata=df.values,
                showlegend=False,
            )
            fig.add_trace(
                bar,
                row=row,
                col=col,
            )

    for i in range(1, n_plots + 1):
        fig["layout"]["xaxis{}".format(i)]["title"] = "(n_samples, n_features)"
        fig["layout"]["yaxis{}".format(i)]["title"] = "Speedup"

    fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=10)))
    fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(size=10)))
    fig.update_annotations(font_size=10)
    fig.update_layout(height=n_rows * 300, barmode="group", showlegend=True)
    fig.show()


def clean_results():
    current_path = Path(__file__).resolve().parent
    extensions = [".csv", ".html"]
    files = []
    for extension in extensions:
        files_path = str(current_path / "results/**/*") + extension
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


def _make_clickable(val):
    return '<a href="{}" target="_blank">{}</a>'.format(val, "See")


def print_profiling_links(algo="", versus_lib=""):
    data = _make_dataset(algo, versus_lib)
    for lib in [BASE_LIB, versus_lib]:
        data[f"{lib}_profiling"] = (
            "results/profiling/"
            + f"{lib}_"
            + data["function"]
            + "_"
            + data["hyperparams_digest"]
            + "_"
            + data["dims_digest"]
            + ".html"
        )
        data[f"{lib}_profiling"] = data[f"{lib}_profiling"].apply(_make_clickable)
    qgrid_widget = qgrid.show_grid(data, show_toolbar=True)
    display(qgrid_widget)


class Reporting:
    """
    Runs reporting for specified estimators.
    """

    def __init__(self, estimators_names=[], versus_libs=[], config_file=""):
        self.estimators_names = estimators_names
        self.versus_libs = versus_libs
        self.config_file = config_file

    def _load_estimators(self):
        with open(self.config_file, "r") as config_file:
            config = yaml.full_load(config_file)
        estimators = config["estimators"]
        if self.estimators_names:
            estimators = {name: estimators[name] for name in self.estimators_names}
        return estimators

    def run(self):
        estimators = self._load_estimators()
        for params in estimators.values():
            report = Report(params, self.versus_libs)
            report.run()


def _compute_ratio_stdev(A, B):
    return A / B


BASE_LIB = "sklearn"


class Report:
    """
    Runs reporting for one estimator.
    """

    def __init__(self, estimator_params={}, versus_libs=[], mode="all"):
        self.estimator_params = estimator_params
        self.versus_libs = versus_libs
        self.mode = mode

    def _make_table_dataset(self):
        libs_files = [
            f"results/{lib}_{self.estimator_name}.csv" for lib in self.versus_libs
        ]
        base_file = f"results/{BASE_LIB}_{self.estimator_name}.csv"
        base_df = pd.read_csv(base_file)
        libs_dfs = [pd.read_csv(file) for file in libs_files]
        merge_cols = [
            "mean",
            "stdev",
            "hyperparams_digest",
            "dims_digest",
            *compare_cols,
        ]
        merged_df = base_df
        for df in libs_dfs:
            lib = df["lib"].values[0]
            merged_df = merged_df.merge(
                df, how="right", on=merge_on_cols, suffixes=["", f"_{lib}"]
            )
            merged_df[f"speedup_{lib}"] = merged_df["mean"] / merged_df[f"mean_{lib}"]
            merged_df[f"stdev_speedup_{lib}"] = _compute_ratio_stdev(
                merged_df["stdev"], merged_df[f"mean_{lib}"]
            )

        numeric_cols = merged_df.select_dtypes(include=["float64"]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].round(4)
        print(merged_df.columns)

        return merged_df

    def _print_table(self):
        data = self._make_table_dataset()
        data[f"{BASE_LIB}_profiling"] = (
            "results/profiling/"
            + f"{BASE_LIB}_"
            + data["function"]
            + "_"
            + data["hyperparams_digest"]
            + "_"
            + data["dims_digest"]
            + ".html"
        )
        data[f"{BASE_LIB}_profiling"] = (
            "<a href='"
            + data[f"{BASE_LIB}_profiling"]
            + "'"
            + " target='_blank'>See</a>"
        )
        for lib in self.versus_libs:
            data[f"{lib}_profiling"] = (
                "results/profiling/"
                + f"{lib}_"
                + data["function"]
                + "_"
                + data["hyperparams_digest"]
                + "_"
                + data["dims_digest"]
                + ".html"
            )
            data[f"{lib}_profiling"] = (
                "<a href='"
                + data[f"{lib}_profiling"]
                + "'"
                + " target='_blank'>See</a>"
            )
        qgrid_widget = qgrid.show_grid(data, show_toolbar=True)
        display(qgrid_widget)

    def _plot(self):
        return self

    def run(self):
        self._print_table()
        self._plot()
