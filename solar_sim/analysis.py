from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .body import Body
from .constants import SUN_NAME

if TYPE_CHECKING:
    from .simulation import SimulationResults


def total_energy_components(bodies: Iterable[Body], G: float) -> tuple[float, float, float]:
    """
    Return (kinetic, potential, total) energy for the full system.
    Potential energy is computed pairwise to avoid double counting.
    """
    bodies = list(bodies)

    kinetic = sum(body.kinetic_energy() for body in bodies)

    potential = 0.0
    for i in range(len(bodies) - 1):
        for j in range(i + 1, len(bodies)):
            separation = bodies[j].position - bodies[i].position
            distance = float(np.linalg.norm(separation))
            potential -= G * bodies[i].mass * bodies[j].mass / distance

    total = kinetic + potential
    return kinetic, potential, total


@dataclass
class OrbitalPeriodDetector:
    """
    Detect orbital periods by watching for an upward crossing of the
    positive x-axis in heliocentric coordinates.

    Since each planet starts on the positive x-axis, the first such
    crossing after t=0 corresponds to one full orbit.
    """

    target_names: list[str]
    central_body_name: str = SUN_NAME

    def __post_init__(self) -> None:
        self.periods: dict[str, float] = {}
        self.previous_relative_positions: dict[str, np.ndarray] = {}
        self.previous_time: float | None = None

    def initialise(self, bodies: Iterable[Body], time: float) -> None:
        bodies_by_name = {body.name.lower(): body for body in bodies}
        if self.central_body_name.lower() not in bodies_by_name:
            raise ValueError(f"Central body '{self.central_body_name}' not found.")

        central_position = bodies_by_name[self.central_body_name.lower()].position

        self.previous_relative_positions = {}
        for name in self.target_names:
            body = bodies_by_name[name.lower()]
            self.previous_relative_positions[name.lower()] = body.position - central_position

        self.previous_time = time

    def update(self, bodies: Iterable[Body], time: float) -> None:
        if self.previous_time is None:
            self.initialise(bodies, time)
            return

        bodies_by_name = {body.name.lower(): body for body in bodies}
        central_position = bodies_by_name[self.central_body_name.lower()].position

        current_relative_positions: dict[str, np.ndarray] = {}
        for name in self.target_names:
            body = bodies_by_name[name.lower()]
            current_relative_positions[name.lower()] = body.position - central_position

        dt = time - self.previous_time

        for name in self.target_names:
            key = name.lower()
            if key in self.periods:
                continue

            previous = self.previous_relative_positions[key]
            current = current_relative_positions[key]

            crossed_positive_x_axis = (
                previous[1] < 0.0 <= current[1] and current[0] > 0.0
            )

            if crossed_positive_x_axis:
                y0 = previous[1]
                y1 = current[1]

                if abs(y1 - y0) > 0.0:
                    fraction = (0.0 - y0) / (y1 - y0)
                else:
                    fraction = 1.0

                crossing_time = self.previous_time + fraction * dt

                # Ignore any pathological near-zero crossing.
                if crossing_time > 0.05:
                    self.periods[key] = crossing_time

        self.previous_relative_positions = current_relative_positions
        self.previous_time = time

    @property
    def all_found(self) -> bool:
        return len(self.periods) == len(self.target_names)


@dataclass
class PeriodErrorSummary:
    """Summary statistics for simulated orbital periods against a reference set."""

    percent_errors: dict[str, float]
    mean_abs_error: float
    max_abs_error: float
    missing_names: list[str]


def summarise_period_errors(
    simulated_periods: dict[str, float],
    reference_periods: dict[str, float],
) -> PeriodErrorSummary:
    percent_errors: dict[str, float] = {}
    absolute_errors: list[float] = []
    missing_names: list[str] = []

    for name, reference_period in reference_periods.items():
        simulated_period = simulated_periods.get(name)

        if simulated_period is None:
            percent_errors[name] = float("nan")
            missing_names.append(name)
            continue

        percent_error = 100.0 * (simulated_period - reference_period) / reference_period
        percent_errors[name] = percent_error
        absolute_errors.append(abs(percent_error))

    if absolute_errors:
        mean_abs_error = float(np.mean(absolute_errors))
        max_abs_error = float(np.max(absolute_errors))
    else:
        mean_abs_error = float("nan")
        max_abs_error = float("nan")

    return PeriodErrorSummary(
        percent_errors=percent_errors,
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        missing_names=missing_names,
    )

@dataclass
class AlignmentEvent:
    """A single contiguous planetary-alignment episode."""

    start_time: float
    peak_time: float
    end_time: float
    duration_years: float
    mean_angle_deg: float
    max_deviation_deg: float


def wrapped_angle_difference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Return the wrapped angular difference angle - reference in [-pi, pi].
    """
    return np.arctan2(np.sin(angle - reference), np.cos(angle - reference))


def arithmetic_mean_angle(angles_rad: np.ndarray) -> float:
    """
    Return the arithmetic mean angle after unwrapping around the largest gap.

    This follows the project wording more literally than taking the angle of
    the mean unit vector, while still handling wrap-around near 0/360 degrees.
    """
    angles = np.mod(np.asarray(angles_rad, dtype=float), 2.0 * np.pi)

    if angles.size == 0:
        raise ValueError("At least one angle is required to compute a mean angle.")

    if angles.size == 1:
        return float(angles[0])

    sorted_angles = np.sort(angles)
    wrapped_angles = np.concatenate([sorted_angles, [sorted_angles[0] + 2.0 * np.pi]])
    gaps = np.diff(wrapped_angles)
    split_index = int(np.argmax(gaps))

    unwrapped_angles = np.concatenate(
        [
            sorted_angles[split_index + 1 :],
            sorted_angles[: split_index + 1] + 2.0 * np.pi,
        ]
    )

    return float(np.mod(np.mean(unwrapped_angles), 2.0 * np.pi))


def alignment_metrics_from_relative_positions(
    relative_positions: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Return the mean angle, per-body deviations, and maximum absolute deviation
    for one alignment snapshot.
    """
    angles_rad = np.mod(
        np.arctan2(relative_positions[:, 1], relative_positions[:, 0]),
        2.0 * np.pi,
    )
    mean_angle_rad = arithmetic_mean_angle(angles_rad)
    deviations_rad = wrapped_angle_difference(angles_rad, mean_angle_rad)
    max_abs_deviation_rad = float(np.max(np.abs(deviations_rad)))

    return mean_angle_rad, deviations_rad, max_abs_deviation_rad


def alignment_metrics_from_bodies(
    bodies: Iterable[Body],
    target_names: list[str],
    central_body_name: str = SUN_NAME,
) -> tuple[float, np.ndarray, float]:
    """
    Return the alignment metrics for the selected bodies at one instant.
    """
    bodies_by_name = {body.name.lower(): body for body in bodies}
    central_key = central_body_name.lower()

    if central_key not in bodies_by_name:
        raise ValueError(f"Central body '{central_body_name}' not found.")

    central_position = bodies_by_name[central_key].position
    relative_positions = np.array(
        [
            bodies_by_name[name.lower()].position - central_position
            for name in target_names
        ],
        dtype=float,
    )

    return alignment_metrics_from_relative_positions(relative_positions)


def compute_alignment_series(
    results: "SimulationResults",
    target_names: list[str],
    central_body_name: str = SUN_NAME,
) -> dict[str, np.ndarray]:
    """
    Compute the mean-angle alignment diagnostics for a set of planets.

    The alignment criterion is based on all selected planets lying within
    some angular threshold of the arithmetic mean angle.
    """
    central_key = central_body_name.lower()
    central_positions = results.positions_history[central_key]

    target_keys = [name.lower() for name in target_names]
    relative_positions = np.stack(
        [
            results.positions_history[name] - central_positions
            for name in target_keys
        ],
        axis=1,
    )  # shape: (n_times, n_bodies, 2)

    n_times = relative_positions.shape[0]
    n_bodies = relative_positions.shape[1]

    angles_rad = np.zeros((n_times, n_bodies), dtype=float)
    mean_angle_rad = np.zeros(n_times, dtype=float)
    deviations_rad = np.zeros((n_times, n_bodies), dtype=float)
    max_abs_deviation_rad = np.zeros(n_times, dtype=float)

    for i in range(n_times):
        mean_angle, deviations, max_deviation = alignment_metrics_from_relative_positions(
            relative_positions[i]
        )
        angles_rad[i] = np.mod(
            np.arctan2(relative_positions[i, :, 1], relative_positions[i, :, 0]),
            2.0 * np.pi,
        )
        mean_angle_rad[i] = mean_angle
        deviations_rad[i] = deviations
        max_abs_deviation_rad[i] = max_deviation

    return {
        "times": results.times.copy(),
        "angles_rad": angles_rad,
        "mean_angle_rad": mean_angle_rad,
        "deviations_rad": deviations_rad,
        "max_abs_deviation_rad": max_abs_deviation_rad,
        "max_abs_deviation_deg": np.degrees(max_abs_deviation_rad),
    }


def detect_alignment_events(
    times: np.ndarray,
    mean_angle_rad: np.ndarray,
    max_abs_deviation_rad: np.ndarray,
    threshold_deg: float,
) -> list[AlignmentEvent]:
    """
    Detect contiguous time intervals during which all planets remain within
    threshold_deg of the arithmetic mean angle.
    """
    threshold_rad = np.deg2rad(threshold_deg)
    aligned = max_abs_deviation_rad <= threshold_rad

    events: list[AlignmentEvent] = []
    in_event = False
    start_index = 0

    for i, is_aligned in enumerate(aligned):
        if is_aligned and not in_event:
            in_event = True
            start_index = i

        event_ends_here = in_event and (
            (not is_aligned) or (i == len(aligned) - 1)
        )

        if event_ends_here:
            if is_aligned and i == len(aligned) - 1:
                end_index = i
            else:
                end_index = i - 1

            window = slice(start_index, end_index + 1)
            local_deviation = max_abs_deviation_rad[window]
            local_peak_offset = int(np.argmin(local_deviation))
            peak_index = start_index + local_peak_offset

            mean_angle_deg = (np.degrees(mean_angle_rad[peak_index]) + 360.0) % 360.0
            max_deviation_deg = float(np.degrees(max_abs_deviation_rad[peak_index]))

            events.append(
                AlignmentEvent(
                    start_time=float(times[start_index]),
                    peak_time=float(times[peak_index]),
                    end_time=float(times[end_index]),
                    duration_years=float(times[end_index] - times[start_index]),
                    mean_angle_deg=float(mean_angle_deg),
                    max_deviation_deg=max_deviation_deg,
                )
            )

            in_event = False

    return events


def alignment_intervals(events: list[AlignmentEvent]) -> np.ndarray:
    """
    Return the intervals between consecutive peak-alignment times.
    """
    if len(events) < 2:
        return np.array([], dtype=float)

    peak_times = np.array([event.peak_time for event in events], dtype=float)
    return np.diff(peak_times)