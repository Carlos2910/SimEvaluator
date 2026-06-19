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

With the existing absolute-path config:

```bash
exp-sim-compare run configs/crimpability_radial_force.yaml
```

With the repo-local study skeleton, after you move the data into `studies/crimpability_radial_force/`:

```bash
exp-sim-compare run configs/crimpability_radial_force_local.yaml
```

Each simulation dataset is its own analysis unit, so outputs are written next to the simulation files that produced them.

## Plot Selected Simulations

```bash
exp-sim-compare plot-selected configs/crimpability_radial_force.yaml
```

This uses `selected_best_simulations_total_force.csv` by default. With a study-local config, grouped figures are written to `studies/<study_name>/selected_plots/`.

## Study Structure

The recommended structure is:

```text
studies/
  crimpability_radial_force/
    experimental/
      W6-AP.txt
      W6-CEEP.txt
      ...
    simulations/
      sim_raw_data_revision/
        sim-W6-AP-Node1.xlsx
        ...
        analysis/
          metrics_by_file.csv
          detected_outliers.csv
          selection_summary_total_force.csv
          interpolated_curves/
          figures/
      sim_raw_data2/
        sim-W6-AP-Node1.xlsx
        ...
        analysis/
          metrics_by_file.csv
          detected_outliers.csv
          selection_summary_total_force.csv
          interpolated_curves/
          figures/
    comparison/
      simulation_dataset_comparison_by_channel.csv
      simulation_dataset_best_by_channel.csv
      simulation_dataset_comparison_by_channel_branch.csv
      simulation_dataset_best_by_channel_branch.csv
      simulation_dataset_comparison_total_force.csv
      selected_best_simulations_total_force.csv
      selected_best_simulations_total_force_by_branch.csv
      sim_results_comparison_total_force.csv
    selected_plots/
      WY-AP_total_force_selected.png
      WY-CEEP_total_force_selected.png
```

Future studies follow the same pattern:

```text
studies/
  axial_compression/
    experimental/
    simulations/
      sim_model_1/
        analysis/
      sim_model_2/
        analysis/
    comparison/
    selected_plots/
```

## Analysis vs Comparison

Per-dataset analysis always runs for each simulation dataset:

```text
studies/<study>/simulations/<dataset>/analysis/
```

Cross-dataset comparison is a multi-dataset feature. It runs only when configured and meaningful:

```yaml
comparison:
  enabled: auto   # auto, true, false
```

- `auto`: run comparison only if at least two simulation datasets exist.
- `true`: require at least two simulation datasets, otherwise raise an error.
- `false`: skip comparison.

Selected plots behave similarly:

```yaml
selected_plot:
  enabled: auto   # auto, true, false
```

- `auto`: plot only if the selected-cases CSV exists.
- `true`: require the selected-cases CSV.
- `false`: skip selected plots.

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
