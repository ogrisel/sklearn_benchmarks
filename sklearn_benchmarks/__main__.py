import yaml
import pandas as pd
from sklearn_benchmarks.core import Algorithm, BenchmarkRunner

def main():
  with open('sklearn_benchmarks/config.yaml', 'r') as file:
    config = yaml.full_load(file)
  
  algorithms = [Algorithm(**params) for params in config['algorithms']]
  datasets = config['datasets']

  for algorithm in algorithms:
    benchmark_runner = BenchmarkRunner(**datasets[algorithm.name])
    results = pd.DataFrame.from_dict(benchmark_runner.run(algorithm))
    results.to_csv(f'sklearn_benchmarks/benchmarks/daal4py/{algorithm.name}.csv', mode='w+')

if __name__ == '__main__':
  main()