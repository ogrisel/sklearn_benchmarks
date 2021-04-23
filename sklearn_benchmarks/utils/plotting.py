import itertools
import re

from sklearn_benchmarks.config import DEFAULT_COMPARE_COLS


def _gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def _order_columns(columns):
    bottom_cols = DEFAULT_COMPARE_COLS

    def order_func(col):
        for bottom_col in bottom_cols:
            pattern = re.compile(f"^({bottom_col})\_.*")
            if pattern.search(col):
                return 1
        return -1

    return sorted(columns, key=lambda col: order_func(col))


def _make_hover_template(df):
    columns = _order_columns(df.columns)
    template = ""
    for index, name in enumerate(columns):
        template += "%s: <b>%%{customdata[%i]}</b><br>" % (name, index)
    template += "<extra></extra>"
    return template
