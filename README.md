# sklearn_benchmarks
> Comparative benchmarking tool for scikit-learn

## Supported third party libraries
- [ ] [daal4py](https://github.com/IntelPython/daal4py) by Intel
- [ ] [cuML](https://github.com/rapidsai/cuml) by NVIDIA
- [ ] [ONNX Runtime](https://github.com/microsoft/onnxruntime) by Microsoft

## Installation

```bash
$ conda install -c conda-forge sklearn_benchmarks
```

## Usage

```
Usage: sklbench [OPTIONS]

Options:
  --config, --c <TEXT>...
                                  Config file path. Example: --c config.yaml [required]

  --help                          Show this message and exit.
```

### Help

To get some help, run:

```
$ sklbench --help
```

### Tests

To run tests:

```
$ pytest tests
```
