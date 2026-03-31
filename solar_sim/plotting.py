from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

if TYPE_CHECKING:
    from .simulation import SimulationResults


def _finalise_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_orbits(
    results: "SimulationResults",
    title: str = "Orbital paths",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, positions in results.positions_history.items():
        colour = results.body_colours.get(name, None)
        label = name.capitalize()

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            label=label,
            color=colour,
            linewidth=1.6,
        )

        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            color=colour,
            s=18,
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def animate_orbits(
    results: "SimulationResults",
    title: str = "Orbital animation",
    interval: int = 20,
    trail_length: int | None = 150,
    frame_stride: int = 1,
) -> tuple[plt.Figure, FuncAnimation]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    frame_indices = list(range(0, len(results.times), frame_stride))
    if frame_indices[-1] != len(results.times) - 1:
        frame_indices.append(len(results.times) - 1)

    all_positions = np.vstack(list(results.positions_history.values()))
    max_extent = float(np.max(np.abs(all_positions)))
    axis_limit = max(1.0, max_extent * 1.1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.grid(True, alpha=0.25)

    line_artists: dict[str, plt.Line2D] = {}
    point_artists: dict[str, plt.Line2D] = {}

    for name, positions in results.positions_history.items():
        colour = results.body_colours.get(name, None)
        label = name.capitalize()

        line_artist, = ax.plot([], [], color=colour, linewidth=1.5, alpha=0.85, label=label)
        point_artist, = ax.plot([], [], marker="o", color=colour, markersize=6, linestyle="None")

        line_artists[name] = line_artist
        point_artists[name] = point_artist

        if positions.size > 0:
            ax.scatter(positions[0, 0], positions[0, 1], color=colour, s=14, alpha=0.35)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="best")

    def update(frame_number: int) -> list[object]:
        frame_index = frame_indices[frame_number]
        updated_artists: list[object] = [time_text]

        for name, positions in results.positions_history.items():
            start_index = 0
            if trail_length is not None:
                start_index = max(0, frame_index - trail_length)

            trail = positions[start_index : frame_index + 1]
            current = positions[frame_index]

            line_artists[name].set_data(trail[:, 0], trail[:, 1])
            point_artists[name].set_data([current[0]], [current[1]])

            updated_artists.append(line_artists[name])
            updated_artists.append(point_artists[name])

        time_text.set_text(f"t = {results.times[frame_index]:.2f} years")
        return updated_artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=False,
        repeat=True,
    )

    fig.tight_layout()
    return fig, animation


def plot_period_comparison(
    simulated_periods: dict[str, float],
    actual_periods: dict[str, float],
    title: str = "Simulated vs actual orbital periods",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    names = list(actual_periods.keys())
    actual = np.array([actual_periods[name] for name in names], dtype=float)
    simulated = np.array(
        [simulated_periods.get(name, np.nan) for name in names],
        dtype=float,
    )

    x = np.arange(len(names), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, actual, width=width, label="Actual", color="tab:blue", alpha=0.85)
    ax.bar(
        x + width / 2,
        simulated,
        width=width,
        label="Simulated",
        color="tab:orange",
        alpha=0.85,
    )

    ax.set_title(title)
    ax.set_ylabel("Orbital period [years]")
    ax.set_xticks(x, [name.capitalize() for name in names], rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_energy_history(
    results: "SimulationResults",
    title: str = "System energy over time",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(results.energy_times, results.kinetic_energy, label="Kinetic", linewidth=1.6)
    ax.plot(results.energy_times, results.potential_energy, label="Potential", linewidth=1.6)
    ax.plot(results.energy_times, results.total_energy, label="Total", linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_total_energy_error(
    results: "SimulationResults",
    title: str = "Relative total-energy drift",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    initial_total_energy = float(results.total_energy[0])

    if initial_total_energy == 0.0:
        relative_error = np.zeros_like(results.total_energy)
    else:
        relative_error = (results.total_energy - initial_total_energy) / abs(initial_total_energy)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results.energy_times, relative_error, color="tab:red", linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Relative error")
    ax.grid(True, alpha=0.25)

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_total_energy_comparison(
    results_by_method: dict[str, "SimulationResults"],
    title: str = "Experiment 2: total energy vs time",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, results in results_by_method.items():
        ax.plot(
            results.energy_times,
            results.total_energy,
            label=method_name,
            linewidth=1.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Total energy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_absolute_relative_energy_drift_comparison(
    results_by_method: dict[str, "SimulationResults"],
    title: str = "Experiment 2: absolute relative total-energy drift",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, results in results_by_method.items():
        initial_total_energy = float(results.total_energy[0])

        if initial_total_energy == 0.0:
            drift = np.zeros_like(results.total_energy)
        else:
            drift = np.abs((results.total_energy - initial_total_energy) / abs(initial_total_energy))

        drift = np.maximum(drift, 1.0e-18)
        ax.semilogy(
            results.energy_times,
            drift,
            label=method_name,
            linewidth=1.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel(r"$|E - E_0| / |E_0|$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_alignment_deviation(
    times: np.ndarray,
    max_abs_deviation_deg: np.ndarray,
    threshold_deg: float,
    event_times: np.ndarray | None = None,
    title: str = "Experiment 4: maximum angular deviation from mean angle",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(times, max_abs_deviation_deg, linewidth=1.5, label="Max deviation from mean angle")
    ax.axhline(
        threshold_deg,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold_deg:.1f}° threshold",
    )

    if event_times is not None and len(event_times) > 0:
        ax.scatter(
            event_times,
            np.full_like(event_times, threshold_deg),
            s=22,
            color="tab:green",
            zorder=3,
            label="Detected events",
        )

    ax.set_title(title)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Maximum angular deviation [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_alignment_intervals(
    intervals_years: np.ndarray,
    title: str = "Experiment 4: intervals between consecutive alignments",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 5))

    if len(intervals_years) == 0:
        ax.text(
            0.5,
            0.5,
            "Fewer than two events found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        event_numbers = np.arange(1, len(intervals_years) + 1, dtype=int)
        ax.plot(event_numbers, intervals_years, marker="o", linewidth=1.6)
        ax.set_xlabel("Interval number")
        ax.set_ylabel("Interval [years]")
        ax.grid(True, alpha=0.25)

    ax.set_title(title)

    _finalise_figure(fig, save_path)
    return fig, ax


def plot_alignment_threshold_counts(
    thresholds_deg: list[float],
    event_counts: list[int],
    title: str = "Experiment 4: number of detected alignments vs threshold",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(thresholds_deg, event_counts, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Threshold [deg]")
    ax.set_ylabel("Number of detected events")
    ax.grid(True, alpha=0.25)

    _finalise_figure(fig, save_path)
    return fig, ax


__all__ = [
    "animate_orbits",
    "plot_alignment_deviation",
    "plot_alignment_intervals",
    "plot_alignment_threshold_counts",
    "plot_absolute_relative_energy_drift_comparison",
    "plot_energy_history",
    "plot_orbits",
    "plot_period_comparison",
    "plot_total_energy_comparison",
    "plot_total_energy_error",
]