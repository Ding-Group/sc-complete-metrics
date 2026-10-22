# sc-complete-metrics

This repository contains the complete implementations of 80 metrics used the manuscript "Comprehensive evaluation of single-cell data integration benchmarking".
This library builds off of [scib](https://github.com/theislab/scib) by adding the additional metrics.

## Installation

```bash
pip install .
pip install ".[rpy2]" 
```

### Metrics that require R

`cell_specific_mixing_score` and `ldf_diff` call R through `rpy2`.
`pip` cannot install R itself, so these need an R installation plus the packages below.
The easiest route is the provided conda environment:

```bash
conda env create -f environment.yml
conda activate sc-complete-metrics
```

If you already have R, install the same packages with `BiocManager::install()` / `install.packages()`
and make sure `R_HOME` points to that installation (or that `R` is on your `PATH`) before installing `rpy2`.


## Usage

Import the library of metrics using `import scib`.
For metrics that rely on R, you may need to set the `R_HOME` environment variable.

```python
os.environ["R_HOME"] = "<path/to/R/in/sc-complete-metrics/environment>" 
```

### Running the eight core metrics

To run the eight metrics suggested by our evaluation of single-cell data integration benchmarking, run `metrics_core`:

```python
from scib.metrics import metrics_core
metric_vals = metrics_core(adata, adata_int, batch_key, label_key, embed)
```

To run all the metrics cataloged, run `metrics_all`:

```python
from scib.metrics import metrics_all
metric_vals = metrics_all(adata, adata_int, batch_key, label_key, embed)
```

For full documentation of each metric and `metrics_core`/`metrics_all`, please consult the [wiki](https://github.com/Ding-Group/sc-complete-metrics/wiki).
