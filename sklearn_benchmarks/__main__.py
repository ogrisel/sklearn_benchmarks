import yaml
import os
import pandas as pd
from sklearn_benchmarks.core import Algorithm, BenchmarkRunner
from sklearn_benchmarks.utils import plot

SUPPORTED_ALGOS = ['NearestNeighbors', 'LogisticRegression', 'Ridge']

def main():
  with open('sklearn_benchmarks/config.yaml', 'r') as file:
    config = yaml.full_load(file)
  
  algorithms = [Algorithm(**params) for params in config['algorithms']]
  datasets = config['datasets']

  for algorithm in algorithms:
    if algorithm.name not in SUPPORTED_ALGOS:
      continue

    benchmark_runner = BenchmarkRunner(**datasets[algorithm.name])
    results = pd.DataFrame.from_dict(benchmark_runner.run(algorithm))
    results.to_csv(f'sklearn_benchmarks/benchmarks/daal4py/{algorithm.name}.csv', mode='w+', index=False)
  
  files = os.listdir('sklearn_benchmarks/benchmarks/daal4py')
  for file in files:
    file = 'sklearn_benchmarks/benchmarks/daal4py/' + file
    bench_resuls = pd.read_csv(file)
    plot(bench_resuls)


if __name__ == '__main__':
  main()