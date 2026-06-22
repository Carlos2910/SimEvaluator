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

After you move the data into `studies/crimpability_radial_force/datasets/`:

```bash
exp-sim-compare run studies/crimpability_radial_force/config.yaml
```

Each simulation dataset is its own analysis unit, so outputs are written next to the simulation files that produced them.

## Plot Selected Simulations

```bash
exp-sim-compare plot-selected studies/crimpability_radial_force/config.yaml
```

This uses `selected_best_simulations_total_force.csv` by default. With a study-local config, grouped figures are written to `studies/<study_name>/selected_plots/`.
Selected plots join loading and unloading into one simulation curve by default; set `selected_plot.split_branches: true` to draw branch segments separately.

## Study Structure

The `configs/` folder is only for reusable templates. Active study configs live inside each study folder as `config.yaml`.

The recommended structure is:

```text
studies/
  crimpability_radial_force/
    config.yaml
    datasets/
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
            interpolated_figures/
        sim_raw_data2/
          sim-W6-AP-Node1.xlsx
          ...
          analysis/
            metrics_by_file.csv
            detected_outliers.csv
            selection_summary_total_force.csv
            interpolated_curves/
            figures/
            interpolated_figures/
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
    config.yaml
    datasets/
      experimental/
      simulations/
        sim_model_1/
          analysis/
            figures/
            interpolated_figures/
        sim_model_2/
          analysis/
            figures/
            interpolated_figures/
    comparison/
    selected_plots/
```

## Analysis vs Comparison

Per-dataset analysis always runs for each simulation dataset:

```text
studies/<study>/datasets/simulations/<dataset>/analysis/
```

The per-dataset diagnostics are split into two figure sets:

```text
studies/<study>/datasets/simulations/<dataset>/analysis/figures/
studies/<study>/datasets/simulations/<dataset>/analysis/interpolated_figures/
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

Before interpolation, the cleaned simulation branch is median-filtered along the diameter axis so narrow force spikes do not dominate the paired comparison.

## Tests

```bash
python3 -m pytest
```
