from __future__ import annotations

import csv
import sys
import time
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_sim import BeemanIntegrator, Simulation, load_bodies_from_json
from solar_sim.analysis import (
    AlignmentEvent,
    alignment_intervals,
    alignment_metrics_from_relative_positions,
    detect_alignment_events,
)
from solar_sim.plotting import (
    plot_alignment_deviation,
    plot_alignment_intervals,
    plot_alignment_threshold_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "parameters_solar.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment4"

DT = 1.0 / 1000.0
SIMULATION_YEARS = 2000.0

TARGET_BODIES = ["mercury", "venus", "earth", "mars", "jupiter"]

MAIN_THRESHOLD_DEG = 5.0
THRESHOLDS_DEG = [5.0, 10.0, 15.0, 20.0, 30.0]
INTERVAL_PLOT_THRESHOLDS = [10.0, 15.0]

PLOT_SAMPLE_INTERVAL_DAYS = 5.0
PROGRESS_UPDATES = 10


def remove_constructed_initial_alignment(
    events: list[AlignmentEvent],
) -> tuple[list[AlignmentEvent], bool]:
    """
    Drop the t = 0 alignment caused by the shared positive-x initial setup.
    """
    if events and np.isclose(events[0].start_time, 0.0, atol=1.0e-12):
        return list(events[1:]), True

    return list(events), False


def save_alignment_events_csv(events, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "start_time_years",
                "peak_time_years",
                "end_time_years",
                "duration_years",
                "mean_angle_deg",
                "max_deviation_deg",
            ]
        )

        for event in events:
            writer.writerow(
                [
                    event.start_time,
                    event.peak_time,
                    event.end_time,
                    event.duration_years,
                    event.mean_angle_deg,
                    event.max_deviation_deg,
                ]
            )


def save_threshold_summary_csv(summary_rows: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "threshold_deg",
                "num_events",
                "num_intervals",
                "mean_interval_years",
                "min_interval_years",
                "max_interval_years",
            ]
        )

        for row in summary_rows:
            writer.writerow(
                [
                    row["threshold_deg"],
                    row["num_events"],
                    row["num_intervals"],
                    row["mean_interval_years"],
                    row["min_interval_years"],
                    row["max_interval_years"],
                ]
            )


def print_event_table(events, threshold_deg: float) -> None:
    print()
    print(f"Experiment 4: alignment events for threshold = {threshold_deg:.1f} degrees")
    print("-" * 108)
    print(
        f"{'Event':<8}"
        f"{'Start [yr]':>14}"
        f"{'Peak [yr]':>14}"
        f"{'End [yr]':>14}"
        f"{'Duration [yr]':>18}"
        f"{'Mean angle [deg]':>20}"
        f"{'Peak max dev [deg]':>20}"
    )
    print("-" * 108)

    if not events:
        print("No events detected.")
        print("-" * 108)
        return

    for i, event in enumerate(events, start=1):
        print(
            f"{i:<8}"
            f"{event.start_time:>14.3f}"
            f"{event.peak_time:>14.3f}"
            f"{event.end_time:>14.3f}"
            f"{event.duration_years:>18.4f}"
            f"{event.mean_angle_deg:>20.3f}"
            f"{event.max_deviation_deg:>20.3f}"
        )

    print("-" * 108)


def print_threshold_summary(summary_rows: list[dict[str, float | int]]) -> None:
    print()
    print("Threshold sensitivity summary")
    print("-" * 88)
    print(
        f"{'Threshold [deg]':<18}"
        f"{'Events':>10}"
        f"{'Intervals':>12}"
        f"{'Mean interval [yr]':>22}"
        f"{'Min interval [yr]':>19}"
        f"{'Max interval [yr]':>19}"
    )
    print("-" * 88)

    for row in summary_rows:
        mean_interval = row["mean_interval_years"]
        min_interval = row["min_interval_years"]
        max_interval = row["max_interval_years"]

        mean_text = "---" if np.isnan(mean_interval) else f"{mean_interval:.3f}"
        min_text = "---" if np.isnan(min_interval) else f"{min_interval:.3f}"
        max_text = "---" if np.isnan(max_interval) else f"{max_interval:.3f}"

        print(
            f"{row['threshold_deg']:<18.1f}"
            f"{row['num_events']:>10}"
            f"{row['num_intervals']:>12}"
            f"{mean_text:>22}"
            f"{min_text:>19}"
            f"{max_text:>19}"
        )

    print("-" * 88)


def run_alignment_scan(simulation: Simulation) -> dict[str, np.ndarray]:
    """
    Scan the simulation at every timestep for alignment detection, while
    keeping a coarser sampled series for plotting.
    """
    total_steps = int(ceil(SIMULATION_YEARS / DT))
    plot_sample_every = max(1, int(round((PLOT_SAMPLE_INTERVAL_DAYS / 365.25) / DT)))
    progress_every = max(1, total_steps // PROGRESS_UPDATES)
    central_body = simulation.body_by_name("sun")
    target_bodies = [simulation.body_by_name(name) for name in TARGET_BODIES]

    def current_alignment_metrics() -> tuple[float, np.ndarray, float]:
        relative_positions = np.array(
            [body.position - central_body.position for body in target_bodies],
            dtype=float,
        )
        return alignment_metrics_from_relative_positions(relative_positions)

    mean_angle, _, max_deviation = current_alignment_metrics()

    exact_times = [simulation.time]
    exact_mean_angles = [mean_angle]
    exact_max_deviations = [max_deviation]

    plot_times = [simulation.time]
    plot_max_deviations_deg = [np.degrees(max_deviation)]

    for step in range(1, total_steps + 1):
        simulation.step()

        mean_angle, _, max_deviation = current_alignment_metrics()

        exact_times.append(simulation.time)
        exact_mean_angles.append(mean_angle)
        exact_max_deviations.append(max_deviation)

        if step % plot_sample_every == 0 or step == total_steps:
            plot_times.append(simulation.time)
            plot_max_deviations_deg.append(np.degrees(max_deviation))

        if step % progress_every == 0 or step == total_steps:
            completion = 100.0 * step / total_steps
            print(
                f"  Completed {completion:5.1f}% "
                f"({simulation.time:.1f} / {SIMULATION_YEARS:.1f} years)"
            )

    return {
        "exact_times": np.array(exact_times, dtype=float),
        "exact_mean_angle_rad": np.array(exact_mean_angles, dtype=float),
        "exact_max_abs_deviation_rad": np.array(exact_max_deviations, dtype=float),
        "plot_times": np.array(plot_times, dtype=float),
        "plot_max_abs_deviation_deg": np.array(plot_max_deviations_deg, dtype=float),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running Experiment 4 for {SIMULATION_YEARS:.1f} years with Beeman...")
    print(f"dt = {DT:.6f} years")
    print(
        "Detecting alignments every timestep and sampling the deviation plot "
        f"every {PLOT_SAMPLE_INTERVAL_DAYS:.1f} days."
    )

    start = time.perf_counter()

    bodies = load_bodies_from_json(DATA_FILE)
    simulation = Simulation(
        bodies=bodies,
        integrator=BeemanIntegrator(),
        dt=DT,
    )

    alignment_scan = run_alignment_scan(simulation)

    elapsed = time.perf_counter() - start
    print(f"Simulation finished in {elapsed:.1f} s")

    main_events = detect_alignment_events(
        times=alignment_scan["exact_times"],
        mean_angle_rad=alignment_scan["exact_mean_angle_rad"],
        max_abs_deviation_rad=alignment_scan["exact_max_abs_deviation_rad"],
        threshold_deg=MAIN_THRESHOLD_DEG,
    )
    main_events, removed_main_initial_event = remove_constructed_initial_alignment(
        main_events
    )

    save_alignment_events_csv(
        main_events,
        OUTPUT_DIR / "experiment4_alignment_events_5deg.csv",
    )

    threshold_summary_rows: list[dict[str, float | int]] = []

    for threshold_deg in THRESHOLDS_DEG:
        events = detect_alignment_events(
            times=alignment_scan["exact_times"],
            mean_angle_rad=alignment_scan["exact_mean_angle_rad"],
            max_abs_deviation_rad=alignment_scan["exact_max_abs_deviation_rad"],
            threshold_deg=threshold_deg,
        )
        events, _ = remove_constructed_initial_alignment(events)

        intervals = alignment_intervals(events)

        if len(intervals) == 0:
            mean_interval = float("nan")
            min_interval = float("nan")
            max_interval = float("nan")
        else:
            mean_interval = float(np.mean(intervals))
            min_interval = float(np.min(intervals))
            max_interval = float(np.max(intervals))

        threshold_summary_rows.append(
            {
                "threshold_deg": threshold_deg,
                "num_events": len(events),
                "num_intervals": len(intervals),
                "mean_interval_years": mean_interval,
                "min_interval_years": min_interval,
                "max_interval_years": max_interval,
            }
        )
        
        if threshold_deg in INTERVAL_PLOT_THRESHOLDS:
            plot_alignment_intervals(
                intervals,
                title=(
                    "Experiment 4: intervals between consecutive "
                    f"{threshold_deg:.0f} deg alignments"
                ),
                save_path=OUTPUT_DIR / f"experiment4_alignment_intervals_{int(threshold_deg)}deg.png",
            )


    save_threshold_summary_csv(
        threshold_summary_rows,
        OUTPUT_DIR / "experiment4_threshold_summary.csv",
    )

    main_intervals = alignment_intervals(main_events)
    main_event_times = np.array([event.peak_time for event in main_events], dtype=float)

    deviation_figure, _ = plot_alignment_deviation(
        times=alignment_scan["plot_times"],
        max_abs_deviation_deg=alignment_scan["plot_max_abs_deviation_deg"],
        threshold_deg=MAIN_THRESHOLD_DEG,
        event_times=main_event_times,
        title=(
            "Experiment 4: maximum angular deviation from the mean angle "
            f"({MAIN_THRESHOLD_DEG:.0f} deg criterion)"
        ),
        save_path=OUTPUT_DIR / "experiment4_max_deviation_5deg.png",
    )

    intervals_figure, _ = plot_alignment_intervals(
        main_intervals,
        title=(
            "Experiment 4: intervals between consecutive "
            f"{MAIN_THRESHOLD_DEG:.0f} deg alignments"
        ),
        save_path=OUTPUT_DIR / "experiment4_alignment_intervals_5deg.png",
    )

    threshold_figure, _ = plot_alignment_threshold_counts(
        thresholds_deg=THRESHOLDS_DEG,
        event_counts=[int(row["num_events"]) for row in threshold_summary_rows],
        title="Experiment 4: detected alignment count vs threshold",
        save_path=OUTPUT_DIR / "experiment4_threshold_counts.png",
    )

    if removed_main_initial_event:
        print()
        print(
            "Excluded the artificial t = 0 alignment created by the shared "
            "initial x-axis setup."
        )

    print_event_table(main_events, MAIN_THRESHOLD_DEG)
    print_threshold_summary(threshold_summary_rows)

    if len(main_intervals) > 0:
        print(
            f"\nMean interval between consecutive {MAIN_THRESHOLD_DEG:.0f} deg alignments: "
            f"{np.mean(main_intervals):.3f} years"
        )
    elif len(main_events) == 1:
        print(
            f"\nOnly one {MAIN_THRESHOLD_DEG:.0f} deg alignment event was detected "
            f"in {SIMULATION_YEARS:.0f} years."
        )
    else:
        print(
            f"\nNo {MAIN_THRESHOLD_DEG:.0f} deg alignment events were detected "
            f"in {SIMULATION_YEARS:.0f} years."
        )

    is_interactive_backend = (
        getattr(deviation_figure.canvas, "required_interactive_framework", None)
        is not None
    )
    if not is_interactive_backend:
        plt.close(deviation_figure)
        plt.close(intervals_figure)
        plt.close(threshold_figure)
        print("Non-interactive matplotlib backend detected, so the plot windows were not shown.")
        return

    plt.show()


if __name__ == "__main__":
    main()