import numpy as np
import importlib
import re
from joblib import Memory


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

def old_plot():
    BAR_WIDTH = .25
    merged_df_knn_grouped = merged_df_knn.groupby(['algorithm', 'n_neighbors', 'function'])
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    axes = axes.flatten()
    for (params, df), ax in zip(merged_df_knn_grouped, axes):
        print(params)
        print(df.shape)
        df = df.loc[(df.n_samples + df.n_features).sort_values().index]
        n_jobs_vals = df['n_jobs'].unique()
        n_bars = len(n_jobs_vals)
        for i, val in enumerate(df['n_jobs'].unique()):
            for 
            x = np.arange(n_bars)
            n_bars = len(n_jobs_vals)
            if i % 2 == 0:
                x = x - (BAR_WIDTH / n_bars)
            else:
                x = x + (BAR_WIDTH / n_bars)
            label="n_jobs = %s" % val
            sns.barplot(x="n_samples", y="speedup", hue="n_jobs", data=df, ax=ax)
            #ax.bar(x, height, width=BAR_WIDTH, label=label)
        title = "algo: %s, k: %s, func: %s" % params
        ax.set_title(title)
        ax.set_xticks(x)
        labels = ["%s (%s, %s)" % tuple(row) for row in df[['function', 'n_samples', 'n_features']].values]
        labels = np.unique(labels)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel('Speedup')
        ax.legend()
        print('---')
    fig.tight_layout()