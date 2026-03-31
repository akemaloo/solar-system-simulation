from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_sim import (
    ACTUAL_ORBITAL_PERIODS,
    BeemanIntegrator,
    DEFAULT_DT,
    Simulation,
    load_bodies_from_json,
    save_period_comparison_csv,
)
from solar_sim.analysis import summarise_period_errors
from solar_sim.plotting import plot_orbits, plot_period_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "parameters_solar.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment1"


def print_period_table(simulated_periods: dict[str, float], actual_periods: dict[str, float]) -> None:
    summary = summarise_period_errors(simulated_periods, actual_periods)

    print("\nExperiment 1: Orbital period comparison")
    print("-" * 68)
    print(f"{'Planet':<12}{'Actual [yr]':>14}{'Simulated [yr]':>18}{'Error [%]':>14}")
    print("-" * 68)

    for name, actual in actual_periods.items():
        simulated = simulated_periods.get(name)
        if simulated is None:
            print(f"{name.capitalize():<12}{actual:>14.6f}{'not found':>18}{'---':>14}")
        else:
            percent_error = 100.0 * (simulated - actual) / actual
            print(f"{name.capitalize():<12}{actual:>14.6f}{simulated:>18.6f}{percent_error:>14.4f}")

    print("-" * 68)
    print(f"Mean absolute error: {summary.mean_abs_error:.4f}%")
    print(f"Max absolute error:  {summary.max_abs_error:.4f}%")


def main() -> None:
    bodies = load_bodies_from_json(DATA_FILE)

    simulation = Simulation(
        bodies=bodies,
        integrator=BeemanIntegrator(),
        dt=DEFAULT_DT,
    )

    results = simulation.run(
        duration_years=13.0,
        record_every=20,
        energy_log_every=20,
        detect_periods=True,
        stop_when_periods_complete=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_period_comparison_csv(
        simulated_periods=results.orbital_periods,
        actual_periods=ACTUAL_ORBITAL_PERIODS,
        output_path=OUTPUT_DIR / "experiment1_periods.csv",
    )

    period_figure, _ = plot_period_comparison(
        simulated_periods=results.orbital_periods,
        actual_periods=ACTUAL_ORBITAL_PERIODS,
        title="Experiment 1: Simulated vs actual orbital periods",
        save_path=OUTPUT_DIR / "experiment1_period_comparison.png",
    )

    orbit_figure, _ = plot_orbits(
        results,
        title="Experiment 1: Orbital paths used for period detection",
        save_path=OUTPUT_DIR / "experiment1_orbits.png",
    )

    print_period_table(results.orbital_periods, ACTUAL_ORBITAL_PERIODS)

    is_interactive_backend = getattr(period_figure.canvas, "required_interactive_framework", None) is not None
    if not is_interactive_backend:
        plt.close(period_figure)
        plt.close(orbit_figure)
        print("Non-interactive matplotlib backend detected, so the plot windows were not shown.")
        return

    plt.show()


if __name__ == "__main__":
    main()