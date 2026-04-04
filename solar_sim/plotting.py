from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D

if TYPE_CHECKING:
    from .simulation import SimulationResults


def _finalise_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


def _split_coordinates(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
        raise ValueError("Position arrays must have shape (n_points, 2) or (n_points, 3).")

    x = positions[:, 0]
    y = positions[:, 1]

    if positions.shape[1] == 2:
        z = np.zeros(len(positions), dtype=float)
    else:
        z = positions[:, 2]

    return x, y, z


def _with_alpha(colour: str | tuple[float, float, float] | tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = mcolors.to_rgba(colour)
    return red, green, blue, alpha


def _body_marker_sizes(name: str) -> tuple[float, float]:
    base_sizes = {
        "sun": 16.0,
        "mercury": 4.2,
        "venus": 5.0,
        "earth": 5.4,
        "mars": 4.6,
        "jupiter": 8.8,
    }
    core_size = base_sizes.get(name.lower(), 5.0)
    glow_size = core_size * 2.15
    return glow_size, core_size


def _make_star_field(
    axis_limit: float,
    z_limit: float,
    count: int = 700,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    cos_polar = rng.uniform(-1.0, 1.0, count)
    polar = np.arccos(cos_polar)
    radius = rng.uniform(axis_limit * 1.35, axis_limit * 2.15, count)

    x = radius * np.sin(polar) * np.cos(azimuth)
    y = radius * np.sin(polar) * np.sin(azimuth)
    z = radius * np.cos(polar) * (z_limit / axis_limit)

    brightness = rng.uniform(0.35, 1.0, count)
    sizes = rng.uniform(2.0, 11.0, count) * brightness
    colours = np.column_stack(
        [
            np.full(count, 0.86),
            np.full(count, 0.91),
            np.full(count, 1.0),
            0.18 + 0.55 * brightness,
        ]
    )
    return x, y, z, sizes, colours


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
    x_all, y_all, z_all = _split_coordinates(all_positions)
    axis_limit = max(
        1.0,
        float(
            max(
                np.max(np.abs(x_all)),
                np.max(np.abs(y_all)),
                np.max(np.abs(z_all)),
            )
        )
        * 1.1,
    )
    z_limit = max(axis_limit * 0.35, float(np.max(np.abs(z_all))) * 1.2, 0.2)

    fig = plt.figure(figsize=(8.8, 8.4))
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#02040a")
    ax.set_facecolor("#02040a")
    ax.set_title(title, color="white", pad=12)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-z_limit, z_limit)
    ax.set_box_aspect((1.0, 1.0, max(0.35, z_limit / axis_limit)))
    ax.view_init(elev=24, azim=35)
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp", focal_length=0.92)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
        axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.set_axis_off()

    stars_x, stars_y, stars_z, star_sizes, star_colours = _make_star_field(axis_limit, z_limit)
    ax.scatter(
        stars_x,
        stars_y,
        stars_z,
        s=star_sizes,
        c=star_colours,
        depthshade=False,
        linewidths=0.0,
    )

    glow_line_artists: dict[str, object] = {}
    line_artists: dict[str, object] = {}
    glow_point_artists: dict[str, object] = {}
    point_artists: dict[str, object] = {}

    for name, positions in results.positions_history.items():
        colour = results.body_colours.get(name, None)
        label = name.capitalize()
        glow_size, core_size = _body_marker_sizes(name)
        trail_alpha = 0.16 if name == "sun" else 0.38
        glow_alpha = 0.08 if name == "sun" else 0.15

        glow_line_artist, = ax.plot(
            [],
            [],
            [],
            color=_with_alpha(colour or "white", glow_alpha),
            linewidth=4.0,
            solid_capstyle="round",
        )
        line_artist, = ax.plot(
            [],
            [],
            [],
            color=_with_alpha(colour or "white", trail_alpha),
            linewidth=1.45 if name != "sun" else 1.1,
            solid_capstyle="round",
            label=label,
        )
        glow_point_artist, = ax.plot(
            [],
            [],
            [],
            marker="o",
            color=_with_alpha(colour or "white", 0.18 if name != "sun" else 0.28),
            markersize=glow_size,
            linestyle="None",
        )
        point_artist, = ax.plot(
            [],
            [],
            [],
            marker="o",
            color=_with_alpha(colour or "white", 1.0),
            markeredgecolor=_with_alpha("white", 0.15 if name != "sun" else 0.35),
            markeredgewidth=0.6 if name != "sun" else 1.0,
            markersize=core_size,
            linestyle="None",
        )

        glow_line_artists[name] = glow_line_artist
        line_artists[name] = line_artist
        glow_point_artists[name] = glow_point_artist
        point_artists[name] = point_artist

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#08111f",
        edgecolor=(1.0, 1.0, 1.0, 0.08),
        framealpha=0.32,
    )
    for text in legend.get_texts():
        text.set_color("white")

    time_text = ax.text2D(
        0.02,
        0.965,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=11,
    )

    def update(frame_number: int) -> list[object]:
        frame_index = frame_indices[frame_number]
        updated_artists: list[object] = [time_text]

        for name, positions in results.positions_history.items():
            start_index = 0
            if trail_length is not None:
                start_index = max(0, frame_index - trail_length)

            trail = positions[start_index : frame_index + 1]
            current = positions[frame_index : frame_index + 1]
            trail_x, trail_y, trail_z = _split_coordinates(trail)
            current_x, current_y, current_z = _split_coordinates(current)

            glow_line_artists[name].set_data(trail_x, trail_y)
            glow_line_artists[name].set_3d_properties(trail_z)
            line_artists[name].set_data(trail_x, trail_y)
            line_artists[name].set_3d_properties(trail_z)
            glow_point_artists[name].set_data(current_x, current_y)
            glow_point_artists[name].set_3d_properties(current_z)
            point_artists[name].set_data(current_x, current_y)
            point_artists[name].set_3d_properties(current_z)

            updated_artists.append(glow_line_artists[name])
            updated_artists.append(line_artists[name])
            updated_artists.append(glow_point_artists[name])
            updated_artists.append(point_artists[name])

        time_text.set_text(f"t = {results.times[frame_index]:.2f} years")
        camera_phase = 2.0 * np.pi * frame_number / max(1, len(frame_indices) - 1)
        ax.view_init(
            elev=22.0 + 4.0 * np.sin(0.8 * camera_phase),
            azim=35.0 + frame_number * 0.18,
        )
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