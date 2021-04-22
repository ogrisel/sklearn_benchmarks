import itertools
from pathlib import Path


def gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def make_hover_template(df):
    template = ""
    for index, name in enumerate(df.columns):
        template += "%s: <b>%%{customdata[%i]}</b><br>" % (name, index)
    template += "<extra></extra>"
    return template