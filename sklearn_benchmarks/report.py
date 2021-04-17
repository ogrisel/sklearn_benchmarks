import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qgrid
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    TIME_REPORT_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.plotting import gen_coordinates_grid, make_hover_template


class Reporting:
    """
    Runs reporting for specified estimators.
    """

    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path

    def _print_time_report():
        df = pd.read_csv(str(TIME_REPORT_PATH), index_col="algo")
        return df

    def run(self):
        config = get_full_config(config_file_path=self.config_file_path)
        reporting_config = config["reporting"]
        display(Markdown("## Time report"))
        self._print_time_report()
        estimators = reporting_config["estimators"]
        for name, params in estimators.items():
            params["n_cols"] = reporting_config["n_cols"]
            display(Markdown(f"## {name}"))
            report = Report(params, self.versus)
            report.run()


class Report:
    """
    Runs reporting for one estimator.
    """

    def __init__(
        self, name="", versus=[], split_bar=[], group_by=[], compare=[], n_cols=None
    ):
        self.name = name
        self.versus = versus
        self.split_bar = split_bar
        self.group_by = group_by
        self.compare = compare
        self.n_cols = n_cols

    def _make_dataset(self, speedup_col="mean", stdev_speedup_col="stdev"):
        benchmarking_results_path = str(BENCHMARKING_RESULTS_PATH)
        lib_df = pd.read_csv(
            "%s/%s_%s.csv" % (benchmarking_results_path, lib, self.name)
        )
        skl_df = pd.read_csv(
            "%s/sklearn_%s.csv" % (benchmarking_results_path, self.name)
        )
        lib_suffix = "_" + lib
        merge_cols = [
            speedup_col,
            stdev_speedup_col,
            "hyperparams_digest",
            "dataset_digest",
            *self.compare,
        ]
        merged_df = pd.merge(
            skl_df[merge_cols],
            lib_df[merge_cols],
            on=["hyperparams_digest", "dataset_digest"],
            suffixes=["", lib_suffix],
        )
        # merged_df = merged_df.drop(["hyperparams_digest", "dataset_digest"], axis=1)
        skl_df = skl_df.drop(merge_cols, axis=1)
        merged_df = pd.merge(skl_df, merged_df, left_index=True, right_index=True)

        numeric_cols = merged_df.select_dtypes(include=["float64"]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].round(4)

        skl_time = merged_df[speedup_col]
        lib_time = merged_df[speedup_col + lib_suffix]
        merged_df["speedup"] = skl_time / lib_time

        skl_std = merged_df[stdev_speedup_col]
        lib_std = merged_df[stdev_speedup_col + lib_suffix]
        merged_df["stdev_speedup"] = merged_df["speedup"] * np.sqrt(
            (skl_std / skl_time) ** 2 + (lib_std / lib_time) ** 2
        )

        merged_df["speedup"] = merged_df["speedup"].round(2)
        merged_df["stdev_speedup"] = merged_df["stdev_speedup"].round(2)

        return merged_df

    def _print_table(self):
        data = self._make_dataset()
        data[f"{BASE_LIB}_profiling"] = (
            "results/profiling/"
            + f"{BASE_LIB}_"
            + data["function"]
            + "_"
            + data["hyperparams_digest"]
            + "_"
            + data["dataset_digest"]
            + ".html"
        )
        data[f"{BASE_LIB}_profiling"] = (
            "<a href='"
            + data[f"{BASE_LIB}_profiling"]
            + "'"
            + " target='_blank'>See</a>"
        )
        for lib in self.versus:
            data[f"{lib}_profiling"] = (
                "results/profiling/"
                + f"{lib}_"
                + data["function"]
                + "_"
                + data["hyperparams_digest"]
                + "_"
                + data["dataset_digest"]
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
        merged_df = self._make_dataset()
        merged_df_grouped = merged_df.groupby(self.group_by)

        n_plots = len(merged_df_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = gen_coordinates_grid(n_rows, self.n_cols)

        subplot_titles = []
        for params, _ in merged_df_grouped:
            title = ""
            for index, (name, val) in enumerate(zip(self.group_by, params)):
                title += "%s: %s" % (name, val)
                if index > 0 and index % 3 == 0:
                    title += "<br>"
                elif index != len(list(zip(self.group_by, params))) - 1:
                    title += " - "

            subplot_titles.append(title)

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

        for (row, col), (_, df) in zip(coordinates, merged_df_grouped):
            df = df.sort_values(by=["n_samples", "n_features"])
            df = df.dropna(axis="columns")
            if self.split_bar:
                for split in self.split_bar:
                    split_vals = df[split].unique()
                    for index, split_val in enumerate(split_vals):
                        x = df[["n_samples", "n_features"]][df[split] == split_val]
                        x = [f"({ns}, {nf})" for ns, nf in x.values]
                        y = df["speedup"][df[split] == split_val]
                        bar = go.Bar(
                            x=x,
                            y=y,
                            name="%s: %s" % (split, split_val),
                            marker_color=px.colors.qualitative.Plotly[index],
                            hovertemplate=make_hover_template(
                                df[df[split] == split_val]
                            ),
                            customdata=df[df[split] == split_val].values,
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
                    hovertemplate=make_hover_template(df),
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

    def run(self):
        self._print_table()
        self._plot()
