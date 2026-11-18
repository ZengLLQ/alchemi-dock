"""Example 4: PROTAC ternary complex cooperativity Hook effect and POI degradation kinetics."""

from __future__ import annotations

from alchemi_dock import (
    default_protac_benchmark_specs,
    render_protac_hook_curve,
    simulate_protac_degradation,
)


def main() -> None:
    specs = default_protac_benchmark_specs()
    for key, spec in specs.items():
        res = simulate_protac_degradation(spec, hours=24.0)
        print(
            f"[{key}] alpha={res.curve.estimated_alpha:.2f} | HookPeak={res.curve.peak_ternary_uM:.4f} uM "
            f"@ {res.curve.hook_conc_uM:.4f} uM | Dmax={100.0 * res.dmax_fraction:.1f}% | "
            f"DC50={res.dc50_uM:.4f} uM | MassErr={res.curve.mass_balance_max_err:.2e}"
        )

    print("")
    best_res = simulate_protac_degradation(specs["optimal_rigid_linker"], hours=24.0)
    print(render_protac_hook_curve(best_res))


if __name__ == "__main__":
    main()
