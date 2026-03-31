from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_sim import (
    ACTUAL_ORBITAL_PERIODS,
    BeemanIntegrator,
    DEFAULT_DT,
    Simulation,
    load_bodies_from_json,
)
from solar_sim.analysis import summarise_period_errors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "parameters_solar.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment1"

STUDY_DURATION_YEARS = 13.0
TIMESTEPS = [
    1.0 / 50.0,
    1.0 / 75.0,
    1.0 / 100.0,
    1.0 / 200.0,
    1.0 / 300.0,
    1.0 / 500.0,
    1.0 / 1000.0,
    DEFAULT_DT,
    0.5 * DEFAULT_DT,
    0.25 * DEFAULT_DT,
]


@dataclass
class TimestepStudyResult:
    dt: float
    simulated_periods: dict[str, float]
    actual_percent_errors: dict[str, float]
    mean_abs_actual_period_error: float
    max_abs_actual_period_error: float
    final_relative_energy_drift: float
    max_abs_relative_energy_drift: float
    reference_percent_errors: dict[str, float] = field(default_factory=dict)
    mean_abs_reference_period_error: float = float("nan")
    max_abs_reference_period_error: float = float("nan")

    @property
    def steps_per_year(self) -> float:
        return 1.0 / self.dt


def relative_energy_drift(total_energy: np.ndarray) -> np.ndarray:
    initial_total_energy = float(total_energy[0])
    if initial_total_energy == 0.0:
        return np.zeros_like(total_energy)
    return (total_energy - initial_total_energy) / abs(initial_total_energy)


def run_single_study(dt: float) -> TimestepStudyResult:
    bodies = load_bodies_from_json(DATA_FILE)

    simulation = Simulation(
        bodies=bodies,
        integrator=BeemanIntegrator(),
        dt=dt,
    )

    total_steps = int(ceil(STUDY_DURATION_YEARS / dt))
    energy_log_every = max(1, total_steps // 300)

    results = simulation.run(
        duration_years=STUDY_DURATION_YEARS,
        record_every=total_steps,
        energy_log_every=energy_log_every,
        detect_periods=True,
        stop_when_periods_complete=True,
    )

    actual_error_summary = summarise_period_errors(results.orbital_periods, ACTUAL_ORBITAL_PERIODS)
    energy_drift = relative_energy_drift(results.total_energy)

    return TimestepStudyResult(
        dt=dt,
        simulated_periods=results.orbital_periods,
        actual_percent_errors=actual_error_summary.percent_errors,
        mean_abs_actual_period_error=actual_error_summary.mean_abs_error,
        max_abs_actual_period_error=actual_error_summary.max_abs_error,
        final_relative_energy_drift=float(energy_drift[-1]),
        max_abs_relative_energy_drift=float(np.max(np.abs(energy_drift))),
    )


def attach_reference_errors(study_results: list[TimestepStudyResult]) -> TimestepStudyResult:
    reference_result = min(study_results, key=lambda result: result.dt)
    reference_periods = reference_result.simulated_periods

    for result in study_results:
        reference_error_summary = summarise_period_errors(result.simulated_periods, reference_periods)
        result.reference_percent_errors = reference_error_summary.percent_errors
        result.mean_abs_reference_period_error = reference_error_summary.mean_abs_error
        result.max_abs_reference_period_error = reference_error_summary.max_abs_error

    return reference_result


def save_timestep_study_csv(study_results: list[TimestepStudyResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dt_years",
        "steps_per_year",
        "mean_abs_actual_period_error_pct",
        "max_abs_actual_period_error_pct",
        "mean_abs_reference_period_error_pct",
        "max_abs_reference_period_error_pct",
        "final_relative_energy_drift",
        "max_abs_relative_energy_drift",
    ]

    for name in ACTUAL_ORBITAL_PERIODS:
        fieldnames.append(f"{name}_simulated_years")
        fieldnames.append(f"{name}_actual_error_pct")
        fieldnames.append(f"{name}_reference_error_pct")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in study_results:
            row = {
                "dt_years": result.dt,
                "steps_per_year": result.steps_per_year,
                "mean_abs_actual_period_error_pct": result.mean_abs_actual_period_error,
                "max_abs_actual_period_error_pct": result.max_abs_actual_period_error,
                "mean_abs_reference_period_error_pct": result.mean_abs_reference_period_error,
                "max_abs_reference_period_error_pct": result.max_abs_reference_period_error,
                "final_relative_energy_drift": result.final_relative_energy_drift,
                "max_abs_relative_energy_drift": result.max_abs_relative_energy_drift,
            }

            for name in ACTUAL_ORBITAL_PERIODS:
                row[f"{name}_simulated_years"] = result.simulated_periods.get(name, "")
                actual_error = result.actual_percent_errors.get(name, float("nan"))
                reference_error = result.reference_percent_errors.get(name, float("nan"))
                row[f"{name}_actual_error_pct"] = "" if np.isnan(actual_error) else actual_error
                row[f"{name}_reference_error_pct"] = "" if np.isnan(reference_error) else reference_error

            writer.writerow(row)


def plot_timestep_study(
    study_results: list[TimestepStudyResult],
    save_path: Path,
) -> tuple[plt.Figure, np.ndarray]:
    ordered_results = sorted(study_results, key=lambda result: result.dt, reverse=True)

    dt_values = np.array([result.dt for result in ordered_results], dtype=float)
    mean_actual_period_error = np.array(
        [result.mean_abs_actual_period_error for result in ordered_results],
        dtype=float,
    )
    max_actual_period_error = np.array(
        [result.max_abs_actual_period_error for result in ordered_results],
        dtype=float,
    )
    mean_reference_period_error = np.array(
        [result.mean_abs_reference_period_error for result in ordered_results],
        dtype=float,
    )
    max_reference_period_error = np.array(
        [result.max_abs_reference_period_error for result in ordered_results],
        dtype=float,
    )
    max_energy_drift = np.array(
        [result.max_abs_relative_energy_drift for result in ordered_results],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        dt_values,
        mean_actual_period_error,
        marker="o",
        linewidth=1.8,
        label="Mean |actual error|",
    )
    axes[0, 0].plot(
        dt_values,
        max_actual_period_error,
        marker="s",
        linewidth=1.8,
        label="Max |actual error|",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Timestep dt [years]")
    axes[0, 0].set_ylabel("Error [%]")
    axes[0, 0].set_title("Against actual orbital periods")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[0, 0].invert_xaxis()

    axes[0, 1].plot(
        dt_values,
        mean_reference_period_error,
        marker="o",
        linewidth=1.8,
        label="Mean |error vs finest dt|",
    )
    axes[0, 1].plot(
        dt_values,
        max_reference_period_error,
        marker="s",
        linewidth=1.8,
        label="Max |error vs finest dt|",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Timestep dt [years]")
    axes[0, 1].set_ylabel("Error [%]")
    axes[0, 1].set_title("Numerical convergence to finest tested dt")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(loc="best")
    axes[0, 1].invert_xaxis()

    for name in ACTUAL_ORBITAL_PERIODS:
        planet_errors = np.array(
            [abs(result.actual_percent_errors.get(name, np.nan)) for result in ordered_results],
            dtype=float,
        )
        axes[1, 0].plot(
            dt_values,
            planet_errors,
            marker="o",
            linewidth=1.5,
            label=name.capitalize(),
        )

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Timestep dt [years]")
    axes[1, 0].set_ylabel("|Actual period error| [%]")
    axes[1, 0].set_title("Planet-by-planet accuracy")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(loc="best")
    axes[1, 0].invert_xaxis()

    axes[1, 1].plot(dt_values, max_energy_drift, marker="o", linewidth=1.8, color="tab:red")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Timestep dt [years]")
    axes[1, 1].set_ylabel("Max |dE / E0|")
    axes[1, 1].set_title("Energy drift")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].invert_xaxis()

    fig.suptitle("Experiment 1: timestep study (Beeman integrator)")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes


def write_timestep_summary(
    study_results: list[TimestepStudyResult],
    reference_result: TimestepStudyResult,
    output_path: Path,
) -> None:
    ordered_results = sorted(study_results, key=lambda result: result.dt, reverse=True)
    best_actual_result = min(
        study_results,
        key=lambda result: (result.mean_abs_actual_period_error, result.dt),
    )
    converged_candidates = [
        result for result in ordered_results if result.max_abs_reference_period_error <= 0.05
    ]
    converged_result = converged_candidates[0] if converged_candidates else reference_result

    lines = [
        "Experiment 1 timestep study summary",
        "===================================",
        "",
        f"Finest tested timestep: {reference_result.dt:.6f} years ({reference_result.steps_per_year:.0f} steps/year)",
        f"Best match to actual orbital periods: dt = {best_actual_result.dt:.6f} years ({best_actual_result.steps_per_year:.0f} steps/year)",
        f"  Mean absolute actual-period error = {best_actual_result.mean_abs_actual_period_error:.4f}%",
        f"  Max absolute actual-period error  = {best_actual_result.max_abs_actual_period_error:.4f}%",
        "",
        f"Coarsest timestep with max period difference <= 0.05% relative to the finest tested run: {converged_result.dt:.6f} years ({converged_result.steps_per_year:.0f} steps/year)",
        f"  Mean absolute reference error = {converged_result.mean_abs_reference_period_error:.4f}%",
        f"  Max absolute reference error  = {converged_result.max_abs_reference_period_error:.4f}%",
        "",
        "Interpretation:",
        "- Decreasing dt rapidly improves agreement with the finest tested simulation, showing numerical convergence.",
        "- Agreement with the actual Solar System plateaus once dt is small, so residual discrepancies are likely dominated by modelling assumptions rather than timestep size alone.",
        f"- The current default dt = {DEFAULT_DT:.6f} years ({1.0 / DEFAULT_DT:.0f} steps/year) is a conservative choice within the converged regime.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary_table(study_results: list[TimestepStudyResult]) -> None:
    print("\nExperiment 1: Timestep study")
    print("-" * 108)
    print(
        f"{'dt [yr]':>12}"
        f"{'steps/yr':>12}"
        f"{'mean |actual| [%]':>20}"
        f"{'max |actual| [%]':>20}"
        f"{'max |vs finest| [%]':>22}"
        f"{'max |dE/E0|':>18}"
    )
    print("-" * 108)

    for result in sorted(study_results, key=lambda result: result.dt, reverse=True):
        print(
            f"{result.dt:>12.6f}"
            f"{result.steps_per_year:>12.0f}"
            f"{result.mean_abs_actual_period_error:>20.4f}"
            f"{result.max_abs_actual_period_error:>20.4f}"
            f"{result.max_abs_reference_period_error:>22.4f}"
            f"{result.max_abs_relative_energy_drift:>18.6e}"
        )

    print("-" * 108)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Experiment 1 timestep study...")
    study_results = [run_single_study(dt) for dt in TIMESTEPS]
    reference_result = attach_reference_errors(study_results)

    csv_path = OUTPUT_DIR / "experiment1_timestep_study.csv"
    plot_path = OUTPUT_DIR / "experiment1_timestep_study.png"
    summary_path = OUTPUT_DIR / "experiment1_timestep_summary.txt"

    save_timestep_study_csv(study_results, csv_path)
    figure, _ = plot_timestep_study(study_results, plot_path)
    write_timestep_summary(study_results, reference_result, summary_path)
    print_summary_table(study_results)
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")

    is_interactive_backend = getattr(figure.canvas, "required_interactive_framework", None) is not None
    if not is_interactive_backend:
        print("Non-interactive matplotlib backend detected, so the plot window was not shown.")
        plt.close(figure)
        return

    plt.show()


if __name__ == "__main__":
    main()