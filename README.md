# SimEvaluator: Simulation–Experiment Agreement Pipeline

A Python pipeline for the automated, metric-based evaluation of finite element (FE) simulation datasets against experimental mechanical test curves. SimEvaluator quantifies agreement through error metrics, detects and classifies outlier spikes, ranks competing simulation variants, and generates diagnostic and publication-ready comparison plots.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output Structure](#output-structure)
- [Analysis Workflow](#analysis-workflow)
- [Key Metrics](#key-metrics)
- [Configuration Reference](#configuration-reference)
- [Tests](#tests)
- [Limitations](#limitations)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

SimEvaluator evaluates how well FE simulation force–displacement curves match experimental mechanical test data. It was developed for the analysis of radial-force and crimpability tests of braided medical devices, where multiple simulation datasets — representing different mesh revisions, material parameters, or output nodes — must be compared against experimental curves to identify the best-fitting model.

The pipeline automates the full evaluation workflow:

- Curve alignment in the diameter domain
- Outlier spike detection with a two-tier flagging and exclusion system
- Loading and unloading branch splitting at maximum compression
- Adaptive metric grid selection between simulation-native and experimental grids
- Error metric computation (RMSE, MAE, sMAPE, NRMSE)
- Cross-dataset ranking and best-simulation selection per test case
- Diagnostic and selected-case figure generation

Full implementation details and configuration options are documented in this repository, which is the reference codebase cited in [[PAPER CITATION]].

## Features

- **Diameter-domain alignment**: Converts simulation displacement to diameter by anchoring the simulated minimum diameter to the experimental minimum, ensuring both curves share the same physical reference
- **Two-tier outlier detection**: Rolling Hampel filter (diameter-span window) separates *flagged* (borderline) from *excluded* (strong spike) points — only spikes whose residual exceeds a configurable multiple of the detection threshold are removed; borderline data is never silently discarded
- **Adaptive metric grid**: Automatically selects the sparser curve's native grid as the comparison reference, interpolating the denser curve onto it to avoid resolution-driven metric inflation
- **Branch-resolved metrics**: Computes RMSE, MAE, sMAPE, bias, max absolute error and NRMSE independently for loading and unloading branches as well as the combined weighted curve
- **Multi-dataset ranking**: Ranks all simulation datasets per test case by composite rank score across selected metrics and selects the best-fitting variant per case
- **Publication-ready plots**: Per-simulation diagnostic figures and selected-case overlay plots supporting raw, cleaned, or smoothed native simulation curves
- **Generic study structure**: Any study with experimental and simulation datasets follows the same folder layout — no hardcoded dataset names anywhere in the pipeline

## Repository Structure

```
SimEvaluator/
├── configs/
│   └── study_template.yaml      # Reusable config template for new studies
├── src/
│   └── exp_sim_compare/
│       ├── cli.py               # Command-line entry points
│       ├── pipeline.py          # Main orchestration loop
│       ├── loaders.py           # Experimental and simulation file readers
│       ├── alignment.py         # Diameter-domain alignment
│       ├── outliers.py          # Hampel-based outlier detection
│       ├── interpolation.py     # Grid pairing and metric grid selection
│       ├── metrics.py           # Error metric computation
│       ├── branches.py          # Loading/unloading branch splitting
│       ├── ranking.py           # Cross-dataset ranking and selection
│       ├── plotting.py          # Diagnostic and selected-case figures
│       └── reports.py           # CSV output writers
├── tests/                       # Unit and integration tests (pytest)
├── studies/                     # Local study folders (gitignored)
├── pyproject.toml               # Package metadata and dependencies
├── README.md                    # This documentation
└── .gitignore
```

> **Note**: The `studies/` folder is gitignored because it contains local datasets, generated analyses, and study-specific configuration files. Use `configs/study_template.yaml` as the starting point for new studies.

## Installation

### Prerequisites

- Python ≥ 3.10
- [Conda](https://docs.conda.io/) (recommended) or pip with a virtual environment

### Option A — Conda (recommended)

```bash
conda create -n exp-sim-compare python=3.11 -y
conda activate exp-sim-compare
git clone https://github.com/Carlos2910/SimEvaluator.git
cd SimEvaluator
pip install -e ".[dev]"
```

### Option B — pip with virtual environment

```bash
git clone https://github.com/Carlos2910/SimEvaluator.git
cd SimEvaluator
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Dependencies

| Package | Minimum version | Purpose |
|---|---|---|
| `matplotlib` | ≥ 3.7 | Diagnostic and comparison plots |
| `numpy` | ≥ 1.24 | Numerical array operations |
| `openpyxl` | ≥ 3.1 | Reading simulation `.xlsx` files |
| `pandas` | ≥ 2.0 | Tabular data and CSV output |
| `PyYAML` | ≥ 6.0 | Study configuration parsing |
| `pytest` | ≥ 7.0 | Test suite (dev only) |

## Usage

### Run the full analysis pipeline

```bash
exp-sim-compare run studies/<study_name>/config.yaml
```

This processes every simulation dataset defined in the config, computes metrics, writes per-dataset outputs, and (if multiple datasets exist) runs cross-dataset comparison and ranking.

### Generate selected-case plots

```bash
exp-sim-compare plot-selected studies/<study_name>/config.yaml
```

Reads `selected_best_simulations_total_force.csv` from the comparison folder and produces one overlay plot per test case, showing the experimental curve alongside the best-fitting simulation.

### Example — crimpability/radial-force study

```bash
exp-sim-compare run studies/crimpability_radial_force/config.yaml
exp-sim-compare plot-selected studies/crimpability_radial_force/config.yaml
```

## Input Data Format

### Experimental data

Plain-text files (`.txt`) with two columns: `diameter` (mm) and `force` (N), space- or tab-delimited. One file per test case:

```
diameter    force
6.150       0.00
6.100       1.23
...
```

### Simulation data

Excel files (`.xlsx`) exported from FE post-processing. Required columns:

| Column | Description |
|---|---|
| `disp` | Displacement (mm) |
| `RF1` | Reaction force — component 1 (N) |
| `RF2` | Reaction force — component 2 (N) |
| `RF3` | Reaction force — component 3 (N) |

`total_force = sqrt(RF1² + RF2² + RF3²)` is computed internally.

### Study directory layout

```
studies/
  <study_name>/
    config.yaml                      # Study-local configuration
    datasets/
      experimental/
        W6-AP.txt
        W6-CEEP.txt
        ...
      simulations/
        <dataset_name_1>/
          sim-W6-AP-Node1.xlsx
          sim-W6-AP-Node6.xlsx
          ...
        <dataset_name_2>/
          sim-W6-AP-Node1.xlsx
          ...
```

## Output Structure

Per-dataset outputs are written alongside the simulation files:

```
datasets/simulations/<dataset_name>/analysis/
├── metrics_by_file.csv              # Full metric table (all branches, channels, variants)
├── detected_outliers.csv            # Flagged and excluded points per file
├── selection_summary_total_force.csv
├── figures/                         # Native simulation curve diagnostics
└── interpolated_figures/            # Paired interpolated curve diagnostics
```

Cross-dataset comparison outputs (written when ≥ 2 datasets exist):

```
comparison/
├── simulation_dataset_comparison_by_channel.csv
├── simulation_dataset_comparison_by_channel_branch.csv
├── simulation_dataset_best_by_channel.csv
├── simulation_dataset_best_by_channel_branch.csv
├── simulation_dataset_comparison_total_force.csv
├── selected_best_simulations_total_force.csv
├── selected_best_simulations_total_force_by_branch.csv
└── sim_results_comparison_total_force.csv  # Per-case summary with one column group per dataset

selected_plots/
└── <case>_total_force_selected.png
```

## Analysis Workflow

1. **Load experimental data** — reads diameter–force text files for each test case
2. **Load simulation data** — reads displacement–reaction-force Excel files; computes `total_force`
3. **Diameter alignment** — converts simulation displacement to diameter by matching the simulated minimum diameter to the experimental minimum diameter
4. **Outlier detection** — applies a rolling Hampel filter per branch; classifies points as *flagged* (borderline) or *excluded* (strong spike, residual/threshold ≥ `exclusion_threshold_ratio`)
5. **Branch splitting** — splits each curve at minimum diameter (maximum compression) into loading and unloading segments; the minimum-diameter point is shared by both branches
6. **Metric grid selection** — chooses the sparser curve's grid as the comparison reference (`auto` mode); interpolates the denser curve onto it
7. **Metric computation** — computes RMSE, MAE, sMAPE, bias, max absolute error, NRMSE_peak, and NRMSE_percent for loading, unloading, and combined-weighted branches under three metric variants: `raw`, `outliers_flagged`, `outliers_excluded`
8. **Ranking** — assigns rank scores per metric across all datasets and cases; produces composite rank sums; selects the best simulation per test case
9. **Plotting** — generates per-simulation diagnostic figures and selected-case overlay plots

## Key Metrics

| Metric | Symbol | Description |
|---|---|---|
| Root Mean Square Error | RMSE | Overall force deviation (N) |
| Mean Absolute Error | MAE | Average absolute deviation (N) |
| Symmetric Mean Absolute Percent Error | sMAPE | Relative error bounded [0, 200%] |
| Bias | — | Signed mean deviation; positive = simulation over-predicts |
| Max Absolute Error | — | Worst-case point deviation (N) |
| Normalized RMSE (peak) | NRMSE_peak | RMSE normalised by peak experimental force |
| Normalized RMSE (range) | NRMSE_percent | 100 × RMSE / experimental force range (%) |

All metrics are computed independently for the `loading` branch, `unloading` branch, and `combined_weighted` (concatenated loading + unloading, used for ranking and selection).

## Configuration Reference

A study config is a YAML file derived from `configs/study_template.yaml`. Key sections:

```yaml
outliers:
  window_diameter_span: 0.10      # Hampel window width in mm (diameter domain)
  min_window_points: 21           # Minimum points in window (guards sparse files)
  sigma: 6.0                      # Detection threshold multiplier (× robust σ)
  exclusion_threshold_ratio: 3.0  # residual/threshold ratio required for exclusion

metrics:
  comparison_grid: auto           # auto | experimental | simulation_native

selection:
  metric_variant: outliers_excluded
  channel: total_force
  rank_by: [RMSE, sMAPE, NRMSE_percent]

comparison:
  enabled: auto                   # auto | true | false

selected_plot:
  enabled: auto                   # auto | true | false
  simulation_data:
    - smoothed                    # raw | cleaned | smoothed | interpolated
```

`comparison: auto` runs cross-dataset ranking only when ≥ 2 simulation datasets are present.  
`selected_plot: auto` generates overlay plots only when the selected-cases CSV exists.

## Tests

Run the full test suite (15 tests) with:

```bash
pytest
```

Tests cover branch splitting, configuration path resolution, interpolation and grid selection, metric computation, outlier flagging versus exclusion, and selected-plot curve loading.

## Limitations

- **Force channel**: Currently configured for `total_force = sqrt(RF1² + RF2² + RF3²)`. Other force channels are stored but the selection and ranking workflow targets total force by default.
- **Simulation file format**: Requires `.xlsx` files with `disp`, `RF1`, `RF2`, `RF3` columns. Other FE output formats require a custom loader.
- **Experimental file format**: Expects two-column diameter–force text files. Alternative formats require a custom loader.
- **Study generality**: The pipeline is validated for radial-force crimpability tests. Adaptation to other mechanical test types may require adjustments to the alignment and branch-splitting logic.
- **Memory**: Large simulation files with tens of thousands of output points are supported but processing time scales roughly linearly with file size.

## Citation

If you use SimEvaluator in your research, please cite both the paper and the software:

### Software — BibTeX

```bibtex
@software{aguilar_vega_2026_simevaluator,
  author       = {Aguilar Vega, Carlos},
  title        = {{SimEvaluator: Simulation--Experiment Agreement Pipeline}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Carlos2910/SimEvaluator},
  version      = {0.1.0}
}
```

### Software — APA

```
Aguilar Vega, C. (2026). SimEvaluator: Simulation–experiment agreement pipeline
(Version 0.1.0) [Computer software]. https://github.com/Carlos2910/SimEvaluator
```

### Software — Plain text

```
Aguilar Vega, Carlos. (2026). SimEvaluator: Simulation–Experiment Agreement Pipeline.
Retrieved from https://github.com/Carlos2910/SimEvaluator
```

## Contributing

SimEvaluator was developed for mechanical testing research and is structured to be extended to new study types.

### Ways to contribute

- **Bug reports**: Open a GitHub issue with a minimal reproducible example
- **New study types**: Adapt the loader and alignment modules for other test geometries
- **Additional metrics**: Extend `metrics.py` with domain-specific error measures
- **Documentation**: Improve configuration examples and workflow guides

### Development setup

```bash
git clone https://github.com/Carlos2910/SimEvaluator.git
cd SimEvaluator
conda create -n exp-sim-compare python=3.11 -y
conda activate exp-sim-compare
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**SimEvaluator** — Automated metric-based evaluation of FE simulation accuracy against experimental mechanical test data.
