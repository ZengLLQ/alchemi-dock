"""Integration tests for full pipeline, JSON telemetry, all CLI commands, examples, and 7-bit ASCII."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alchemi_dock.cli import main as cli_main
from alchemi_dock.pipeline import campaign_report_to_dict, run_full_discovery_campaign
from alchemi_dock.visualizer import assert_ascii, render_campaign_dashboard


def test_full_campaign_pipeline_and_json_serialization(tmp_path: Path) -> None:
    report = run_full_discovery_campaign(architecture="gatv2", epochs=12, opt_generations=4, seed=42)
    assert report.library_size == 36
    assert report.scaffold_count >= 5
    assert report.cluster_count >= 4
    assert report.protac_degradation.curve.mass_balance_max_err < 1e-11
    assert report.optimization.best_candidate.mpo_score >= report.optimization.generation_best_mpo[0]

    dashboard = render_campaign_dashboard(report)
    assert_ascii(dashboard)
    assert "ALCHEMIDOCK" in dashboard

    payload = campaign_report_to_dict(report)
    out_file = tmp_path / "campaign.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["library_size"] == 36


@pytest.mark.parametrize(
    "argv",
    [
        ["profile", "--smiles", "CC(=O)Oc1ccccc1C(=O)O"],
        ["screen", "--arch", "gcn", "--epochs", "8", "--seed", "42"],
        ["dock", "--smiles", "Fc1ccc(Nc2ncnc3ccccc23)cc1", "--steps", "40", "--seed", "42"],
        ["protac", "--cooperativity", "6.5", "--linker-length", "10"],
        ["optimize", "--generations", "3", "--population", "8", "--seed", "42"],
        ["benchmark", "--epochs", "8", "--generations", "3", "--seed", "42"],
        ["demo", "--epochs", "8", "--generations", "3", "--seed", "42"],
    ],
)
def test_cli_subcommands_strict_ascii_cp1252(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(argv)
    assert rc == 0
    captured = capsys.readouterr()
    assert_ascii(captured.out)
    # Confirm Windows cp1252 encoding succeeds without UnicodeEncodeError
    captured.out.encode("cp1252", errors="strict")


def test_examples_execute_cleanly() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    scripts = sorted(examples_dir.glob("*.py"))
    assert len(scripts) == 5
    for script in scripts:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc.returncode == 0, f"{script.name} failed:\n{proc.stderr}"
        assert_ascii(proc.stdout)
