"""
The main entry point. Invoke as `sklearn_benchmarks' or `python sklearn_benchmarks'.
"""
import click
import yaml
import time
import pandas as pd
from sklearn_benchmarks.utils.misc import clean_results, convert
from sklearn_benchmarks.benchmark import Benchmark
from sklearn_benchmarks.config import (
    DEFAULT_CONFIG_FILE_PATH,
    TIME_REPORT_PATH,
    prepare_params,
)


@click.command()
@click.option(
    "--append",
    "--a",
    is_flag=True,
    default=False,
    help="Append benchmark results to existing ones.",
)
@click.option(
    "--config",
    "--c",
    type=str,
    default=DEFAULT_CONFIG_FILE_PATH,
    show_default=True,
    help="Path to config file.",
)
def main(append, config):
    if not append:
        clean_results()

    with open(config, "r") as config:
        config = yaml.full_load(config)

    estimators = config["estimators"]

    time_report = pd.DataFrame(columns=["algo", "hour", "min", "sec"])
    t0 = time.perf_counter()
    for name, params in estimators.items():
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = estimators[params["inherit"]]
            params["estimator"] = curr_estimator

        params = prepare_params(params)

        benchmark_estimator = Benchmark(**params)
        t0_ = time.perf_counter()
        benchmark_estimator.run()
        t1_ = time.perf_counter()
        time_report.loc[len(time_report)] = [name, *convert(t1_ - t0_)]
        benchmark_estimator.to_csv()

    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", *convert(t1 - t0)]
    time_report[["hour", "min", "sec"]] = time_report[["hour", "min", "sec"]].astype(
        int
    )
    time_report.to_csv(
        str(TIME_REPORT_PATH),
        mode="w+",
        index=False,
    )


if __name__ == "__main__":
    main()
