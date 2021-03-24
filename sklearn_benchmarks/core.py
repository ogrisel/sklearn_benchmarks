import importlib
import time
import numpy as np
import daal4py.sklearn as d4p

from sklearn_benchmarks.utils import get_bench_func, gen_data

class AlgorithmLoader:
  """
    Adapter to load patched sklearn estimators from libraries
  """
  def __init__(self, lib_name, algo_name):
    self.lib_name = lib_name
    self.algo_name = algo_name
  
  def load(self):
    return self

class BenchmarkTimer:
  """
    Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:
    .. code-block:: python
      timer = BenchmarkTimer(rep=5)
      for _ in timer.benchmark_runs():
        ... do something ...
      print(np.min(timer.timings))
  """

  def __init__(self, reps=1):
    self.reps = reps
    self.timings = []

  def benchmark_runs(self):
    for r in range(self.reps):
      t0 = time.time()
      yield r
      t1 = time.time()
      self.timings.append(t1 - t0)

class Algorithm:
  """
    Wrapper for an algorithm to benchmark
  """
  def __init__(self, algo_module, algo_class, args={}, name=None, bench_func='fit', accepts_labels=True):
    self.name = name
    self.accepts_labels = accepts_labels
    self.algo_module = algo_module
    self.algo_class = algo_class
    self.args = args
    self.bench_func = get_bench_func(bench_func)

  def run(self, data, **override_args): # use algo loader here?
    all_args = {**self.args, **override_args}

    estimator = self.algo_class_(**all_args)
    if self.accepts_labels:
      self.bench_func(estimator, data[0], data[1])
    else:
      self.bench_func(estimator, data[0])
    return estimator

class BenchmarkRunner:
  """
    Wrapper to run an algorithm with multiple dataset sizes and compute speedup of cuml relative to sklearn baseline
  """

  def __init__(self, bench_rows, bench_dims, dataset_name, n_reps=1):
    self.bench_rows = bench_rows
    self.bench_dims = bench_dims
    self.dataset_name = dataset_name
    self.n_reps = n_reps

  def _run_one_size(self, algo, n_samples, n_features, param_overrides={}, dataset_param_overrides={}):

    data = gen_data(self.dataset_name, n_samples, n_features, **dataset_param_overrides)

    # sklearn
    skl_timer = BenchmarkTimer(self.n_reps)
    for rep in skl_timer.benchmark_runs():
      algo.run(data, **param_overrides)
    skl_elapsed = np.min(skl_timer.timings)

    # patched sklearn
    d4p.patch_sklearn()
    d4p_timer = BenchmarkTimer(self.n_reps)
    for rep in d4p_timer.benchmark_runs():
      algo.run(data, **param_overrides)
    d4p_elapsed = np.min(d4p_timer.timings)
    d4p.unpatch_sklearn()

    speedup = skl_elapsed / d4p_elapsed

    print(f"{algo.name} (n_samples={n_samples}, n_features={n_features})"
          f" [skl={skl_elapsed}, d4p={d4p_elapsed} speedup={speedup}]")

    return dict(
      skl_time=skl_elapsed,
      d4p_time=d4p_elapsed,
      speedup=speedup,
      n_samples=n_samples,
      n_features=n_features,
      **param_overrides,
      **dataset_param_overrides
    )

  def run(self, algo, param_overrides={}, dataset_param_overrides={}):
    all_results = []
    for ns in self.bench_rows:
      for nf in self.bench_dims:
        all_results.append(
          self._run_one_size(algo, ns, nf, param_overrides, dataset_param_overrides))
    return all_results
