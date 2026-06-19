from pathlib import Path
from tempfile import TemporaryDirectory

from exp_sim_compare.config import feature_enabled, load_config, study_root
from exp_sim_compare.loaders import simulation_folders
from exp_sim_compare.pipeline import comparison_output_folder, should_run_comparison


def test_study_relative_simulation_paths_resolve_from_study_root():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = {
            "_project_dir": str(tmp_path),
            "study": {"name": "demo", "folder": "studies/demo"},
            "simulation": {
                "folder": "datasets/simulations",
                "datasets": {
                    "sim_a": {"folder": "sim_a"},
                    "sim_b": {"folder": "nested/sim_b"},
                },
            },
        }

        folders = simulation_folders(config)

        assert study_root(config) == (tmp_path / "studies" / "demo").resolve()
        assert folders["sim_a"] == (
            tmp_path / "studies" / "demo" / "datasets" / "simulations" / "sim_a"
        ).resolve()
        assert folders["sim_b"] == (
            tmp_path / "studies" / "demo" / "datasets" / "simulations" / "nested" / "sim_b"
        ).resolve()


def test_comparison_output_folder_uses_study_root():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = {
            "_project_dir": str(tmp_path),
            "study": {"name": "demo", "folder": "studies/demo"},
            "experimental": {"folder": "datasets/experimental"},
            "comparison": {"output_folder": "comparison"},
        }

        assert comparison_output_folder(config) == (
            tmp_path / "studies" / "demo" / "comparison"
        ).resolve()


def test_load_config_infers_project_root_from_study_folder():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        study_dir = root / "studies" / "demo"
        study_dir.mkdir(parents=True)
        config_path = study_dir / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "study:",
                    "  name: demo",
                    "  folder: studies/demo",
                    "experimental:",
                    "  folder: datasets/experimental",
                    "simulation:",
                    "  folder: datasets/simulations",
                    "  datasets:",
                    "    sim_a:",
                    "      folder: sim_a",
                ]
            ),
            encoding="utf-8",
        )

        config = load_config(config_path)
        folders = simulation_folders(config)

        assert Path(config["_project_dir"]) == root.resolve()
        assert study_root(config) == study_dir.resolve()
        assert folders["sim_a"] == (
            study_dir / "datasets" / "simulations" / "sim_a"
        ).resolve()


def test_comparison_auto_true_false_behavior():
    assert should_run_comparison({"comparison": {"enabled": "auto"}}, 2) is True
    assert should_run_comparison({"comparison": {"enabled": "auto"}}, 1) is False
    assert should_run_comparison({"comparison": {"enabled": False}}, 2) is False
    assert should_run_comparison({"comparison": {"enabled": True}}, 2) is True


def test_comparison_true_requires_multiple_datasets():
    try:
        should_run_comparison({"comparison": {"enabled": True}}, 1)
    except ValueError as exc:
        assert "at least two simulation datasets" in str(exc)
    else:
        raise AssertionError("comparison.enabled=true should require at least two datasets")


def test_feature_enabled_auto_true_false():
    assert feature_enabled("auto", auto_condition=True, feature_name="x", require_message="no") is True
    assert feature_enabled("auto", auto_condition=False, feature_name="x", require_message="no") is False
    assert feature_enabled("false", auto_condition=True, feature_name="x", require_message="no") is False
    assert feature_enabled("true", auto_condition=True, feature_name="x", require_message="no") is True

    try:
        feature_enabled("true", auto_condition=False, feature_name="x", require_message="missing")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("enabled=true should enforce its required condition")
