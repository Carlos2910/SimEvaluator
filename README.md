# Experimental Simulation Comparison

Reusable pipeline for comparing experimental test curves with simulation curves.

The first configured workflow reproduces the crimpability/radial-force analysis:

- load experimental `diameter, force` text files,
- load simulation files with `disp`, `RF1`, `RF2`, `RF3`,
- align the simulation minimum diameter to the experimental minimum diameter,
- calculate `total_force = sqrt(RF1^2 + RF2^2 + RF3^2)`,
- detect simulation outliers with a rolling-median/Hampel-like detector,
- split loading and unloading at minimum diameter,
- include the minimum-diameter point in both branches,
- interpolate simulation values onto the experimental diameter grid,
- export paired comparison curves,
- calculate metrics and rank candidate simulations,
- generate diagnostic and selected-case plots.

## Install

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

## Run Crimpability Analysis

```bash
exp-sim-compare run configs/crimpability_radial_force.yaml
```

Outputs are written to each simulation folder's `analysis/` directory and to the configured comparison output folder.

## Plot Selected Simulations

```bash
exp-sim-compare plot-selected configs/crimpability_radial_force.yaml
```

This uses `selected_best_simulations_total_force.csv` by default and writes grouped figures to `analysis/selected_plots/`.

## Main Outputs

```text
analysis/
  metrics_by_file.csv
  detected_outliers.csv
  interpolated_curves/
  figures/
  selection_summary_total_force.csv

comparison output folder/
  simulation_dataset_comparison_by_channel.csv
  simulation_dataset_best_by_channel.csv
  simulation_dataset_comparison_by_channel_branch.csv
  simulation_dataset_best_by_channel_branch.csv
  selected_best_simulations_total_force.csv
  selected_best_simulations_total_force_by_branch.csv
  sim_results_comparison_total_force.csv
```

## Metric Variant Used For Selection

The default selection variant is:

```text
outliers_excluded_interpolated
```

This means metrics are calculated from the exact exported paired curves after simulation outliers are excluded and the cleaned simulation is interpolated onto the experimental diameter grid.

## Tests

```bash
python3 -m pytest
```
