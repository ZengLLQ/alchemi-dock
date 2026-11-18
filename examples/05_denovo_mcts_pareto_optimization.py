"""Example 5: MCTS + NSGA-II Pareto de novo molecular optimization & campaign dashboard."""

from __future__ import annotations

from alchemi_dock import (
    optimize_molecules_mcts_pareto,
    render_campaign_dashboard,
    render_pareto_scatter,
    run_full_discovery_campaign,
)


def main() -> None:
    traj = optimize_molecules_mcts_pareto(
        seed_smiles="Fc1ccc(Nc2ncnc3ccccc23)cc1",
        generations=6,
        population_size=14,
        seed=42,
    )
    print(render_pareto_scatter(traj))
    print("")
    report = run_full_discovery_campaign(architecture="gatv2", epochs=18, opt_generations=5, seed=42)
    print(render_campaign_dashboard(report))


if __name__ == "__main__":
    main()
