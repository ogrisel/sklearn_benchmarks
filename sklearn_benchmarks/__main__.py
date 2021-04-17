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
    get_full_config,
    prepare_params,
)


@click.command()
@click.option(
    "--append",
    "--a",
    is_flag=True,
    required=False,
    default=False,
    help="Append benchmark results to existing ones.",
)
@click.option(
    "--config",
    "--c",
    type=str,
    default=DEFAULT_CONFIG_FILE_PATH,
    help="Path to config file.",
)
@click.option(
    "--profiling_file_type",
    "--pft",
    type=str,
    default="json.gz",
    help="Profiling files type.",
)
def main(append, config, profiling_file_type):
    if not append:
        clean_results()
    config = get_full_config(config)
    benchmarking_config = config["benchmarking"]
    if not "estimators" in benchmarking_config:
        return

    estimators = benchmarking_config["estimators"]

    time_report = pd.DataFrame(columns=["algo", "hour", "min", "sec"])
    t0 = time.perf_counter()
    for name, params in estimators.items():
        # When inherit is set, we fetch params from parent estimator
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = estimators[params["inherit"]]
            params["estimator"] = curr_estimator

        params = prepare_params(params)
        if "random_state" in config:
            params["random_state"] = config["random_state"]
        params["profiling_file_type"] = profiling_file_type
        benchmark_estimator = Benchmark(**params)
        t0_ = time.perf_counter()
        benchmark_estimator.run()
        t1_ = time.perf_counter()
        time_report.loc[len(time_report)] = [name, *convert(t1_ - t0_)]
        benchmark_estimator.to_csv()

    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", *convert(t1 - t0)]
    time_report.to_csv(
        str(TIME_REPORT_PATH),
        mode="w+",
        index=False,
    )


if __name__ == "__main__":
    main()
