from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_sim import (
    BeemanIntegrator,
    DirectEulerIntegrator,
    EulerCromerIntegrator,
    Simulation,
    SimulationResults,
    load_bodies_from_json,
)
from solar_sim.plotting import (
    plot_absolute_relative_energy_drift_comparison,
    plot_energy_history,
    plot_total_energy_comparison,
    plot_total_energy_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "parameters_solar.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment2"

DT = 1.0 / 2000.0
LONG_DURATION_YEARS = 200.0
BEEMAN_ZOOM_YEARS = 20.0
ENERGY_LOG_INTERVAL_YEARS = 0.02


def relative_energy_drift(total_energy: np.ndarray) -> np.ndarray:
    baseline = float(total_energy[0])
    if baseline == 0.0:
        return np.zeros_like(total_energy)
    return (total_energy - baseline) / abs(baseline)


def max_abs_relative_drift(total_energy: np.ndarray) -> float:
    return float(np.max(np.abs(relative_energy_drift(total_energy))))


def final_relative_drift(total_energy: np.ndarray) -> float:
    return float(relative_energy_drift(total_energy)[-1])


def run_one_method(integrator, duration_years: float) -> SimulationResults:
    bodies = load_bodies_from_json(DATA_FILE)

    simulation = Simulation(
        bodies=bodies,
        integrator=integrator,
        dt=DT,
    )

    energy_log_every = max(1, int(round(ENERGY_LOG_INTERVAL_YEARS / DT)))
    record_every = energy_log_every

    results = simulation.run(
        duration_years=duration_years,
        record_every=record_every,
        energy_log_every=energy_log_every,
        detect_periods=False,
    )
    return results


def save_summary_csv(results_by_method: dict[str, SimulationResults], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "duration_years",
                "initial_total_energy",
                "final_total_energy",
                "final_relative_drift",
                "max_abs_relative_drift",
            ]
        )

        for method_name, results in results_by_method.items():
            writer.writerow(
                [
                    method_name,
                    float(results.energy_times[-1]),
                    float(results.total_energy[0]),
                    float(results.total_energy[-1]),
                    final_relative_drift(results.total_energy),
                    max_abs_relative_drift(results.total_energy),
                ]
            )


def print_summary_table(results_by_method: dict[str, SimulationResults]) -> None:
    print("\nExperiment 2: Energy conservation comparison")
    print("-" * 108)
    print(
        f"{'Method':<18}"
        f"{'Duration [yr]':>16}"
        f"{'Initial E':>18}"
        f"{'Final E':>18}"
        f"{'Final dE/E0':>18}"
        f"{'Max |dE/E0|':>18}"
    )
    print("-" * 108)

    for method_name, results in results_by_method.items():
        print(
            f"{method_name:<18}"
            f"{results.energy_times[-1]:>16.2f}"
            f"{results.total_energy[0]:>18.6f}"
            f"{results.total_energy[-1]:>18.6f}"
            f"{final_relative_drift(results.total_energy):>18.6e}"
            f"{max_abs_relative_drift(results.total_energy):>18.6e}"
        )

    print("-" * 108)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_results = {
        "Beeman": run_one_method(BeemanIntegrator(), LONG_DURATION_YEARS),
        "Euler-Cromer": run_one_method(EulerCromerIntegrator(), LONG_DURATION_YEARS),
        "Direct Euler": run_one_method(DirectEulerIntegrator(), LONG_DURATION_YEARS),
    }

    beeman_zoom_results = run_one_method(BeemanIntegrator(), BEEMAN_ZOOM_YEARS)

    save_summary_csv(comparison_results, OUTPUT_DIR / "experiment2_energy_summary.csv")

    comparison_figure, _ = plot_total_energy_comparison(
        comparison_results,
        title=f"Experiment 2: total energy vs time (dt = {DT:.6f} years)",
        save_path=OUTPUT_DIR / "experiment2_total_energy_comparison.png",
    )

    drift_figure, _ = plot_absolute_relative_energy_drift_comparison(
        comparison_results,
        title=f"Experiment 2: absolute relative energy drift (dt = {DT:.6f} years)",
        save_path=OUTPUT_DIR / "experiment2_abs_relative_drift_comparison.png",
    )

    beeman_drift_figure, _ = plot_total_energy_error(
        beeman_zoom_results,
        title=f"Experiment 2: Beeman relative total-energy drift (first {BEEMAN_ZOOM_YEARS:.0f} years)",
        save_path=OUTPUT_DIR / "experiment2_beeman_relative_drift_zoom.png",
    )

    beeman_energy_figure, _ = plot_energy_history(
        beeman_zoom_results,
        title=f"Experiment 2: Beeman energy history (first {BEEMAN_ZOOM_YEARS:.0f} years)",
        save_path=OUTPUT_DIR / "experiment2_beeman_energy_history_zoom.png",
    )

    print_summary_table(comparison_results)

    is_interactive_backend = (
        getattr(comparison_figure.canvas, "required_interactive_framework", None) is not None
    )
    if not is_interactive_backend:
        plt.close(comparison_figure)
        plt.close(drift_figure)
        plt.close(beeman_drift_figure)
        plt.close(beeman_energy_figure)
        print("Non-interactive matplotlib backend detected, so the plot windows were not shown.")
        return

    plt.show()


if __name__ == "__main__":
    main()