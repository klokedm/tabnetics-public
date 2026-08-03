from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analysis" / "diakrino_distribution_scaffold_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("diakrino_distribution_scaffold_probe", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distribution_scaffold_probe_writes_expected_artifacts(tmp_path):
    probe = _load_probe_module()

    report = probe.run_probe(tmp_path)

    assert report["pass"] is True
    assert report["mis_transform_floor"]["skip_fit_discrete"]["discrete_route_rate"] == 1.0
    assert report["mis_transform_floor"]["skip_fit_discrete"]["continuous_false_route_rate"] == 0.0
    assert report["mis_transform_floor"]["cdf_trust"]["high_entropy_route_rate"] == 1.0
    assert report["mis_transform_floor"]["cdf_trust"]["low_entropy_false_route_rate"] == 0.0
    assert report["stability_surrogate"]["with_sidecar_bootstrap_calls"] == 0
    assert report["stability_surrogate"]["without_sidecar_bootstrap_calls"] == report["n_features"]
    assert (tmp_path / "diakrino_distribution_scaffold_probe.json").exists()
    assert (tmp_path / "diakrino_distribution_scaffold_probe.md").exists()
    assert (tmp_path / "diakrino_distribution_stability_weights.csv").exists()
