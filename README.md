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
- choose a fair metric grid from the simulation/experimental resolution,
- export paired comparison curves,
- calculate metrics and rank candidate simulations,
- generate diagnostic and selected-case plots.

## Install

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

## Run Crimpability Analysis

After you create a local study folder and move the data into `studies/crimpability_radial_force/datasets/`:

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
Selected plots can show any combination of native simulation curves:

```yaml
selected_plot:
  simulation_data:
    - raw       # native simulation force
    - cleaned   # native simulation after excluded spikes are removed
    - smoothed  # cleaned native simulation with median smoothing for visual readability
```

Use `interpolated` in the same list when you want to show the processed paired curve used for the exported interpolation audit files.

## Study Structure

The `configs/` folder is only for reusable templates. Active study configs live inside each local study folder as `config.yaml`.

The `studies/` folder is ignored by git because it contains local datasets, generated analyses, selected plots, and study-specific configs. Keep those files on your machine; use `configs/study_template.yaml` as the tracked starting point for new studies.

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
outliers_excluded
```

This uses the cleaned simulation after only strong excluded spikes are removed. The metric grid is controlled by:

```yaml
metrics:
  comparison_grid: auto
```

With `auto`, sparse simulations are compared on the simulation-native diameter grid by interpolating the experiment to the simulation diameters. Denser simulations are compared on the experimental grid. The exported `outliers_excluded_interpolated` curve is still written for auditability, but it is no longer the required default for selecting the best simulation.

Outlier detection separates review from exclusion:

```yaml
outliers:
  window_diameter_span: 0.10
  min_window_points: 21
  exclusion_threshold_ratio: 3.0
```

Points above the Hampel threshold are flagged. The diameter span sets the physical neighborhood, while `min_window_points` prevents sparse files from using a too-small local window. Only stronger points whose residual/threshold ratio is at least `exclusion_threshold_ratio` are removed from cleaned metrics and cleaned plots. Borderline local behavior remains in the data.

## Tests

```bash
python3 -m pytest
```
