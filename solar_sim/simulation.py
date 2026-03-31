from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .analysis import OrbitalPeriodDetector, total_energy_components
from .body import Body
from .constants import EPSILON, G as DEFAULT_G, SUN_NAME

if TYPE_CHECKING:
    from .integrators import Integrator


@dataclass
class SimulationResults:
    """Container for the main outputs produced by a simulation run."""

    times: np.ndarray
    positions_history: dict[str, np.ndarray]
    energy_times: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    orbital_periods: dict[str, float]
    body_colours: dict[str, str]
    method_name: str


class Simulation:
    """N-body solar system simulation."""

    def __init__(
        self,
        bodies: Sequence[Body],
        integrator: Integrator,
        dt: float,
        gravitational_constant: float = DEFAULT_G,
    ) -> None:
        """
        Create a simulation object.

        Parameters
        ----------
        bodies
            Sequence of gravitating bodies to evolve.
        integrator
            Time-integration method object with a step(simulation, dt) method.
        dt
            Timestep in years.
        gravitational_constant
            Gravitational constant in the chosen project units.
        """
        if len(bodies) < 2:
            raise ValueError("The simulation requires at least two bodies.")

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.bodies = list(bodies)
        self.integrator = integrator
        self.dt = float(dt)
        self.G = float(gravitational_constant)

        self.time = 0.0
        self.step_count = 0

        self.update_current_accelerations()
        self.initialise_previous_accelerations()

    def body_by_name(self, name: str) -> Body:
        """Return a body by name, case-insensitively."""
        for body in self.bodies:
            if body.name.lower() == name.lower():
                return body
        raise KeyError(f"No body named '{name}' found.")

    def state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return copies of the simulation state as NumPy arrays.

        Returns
        -------
        positions, velocities, accelerations, previous_accelerations
            Arrays of shape (n_bodies, 2).
        """
        positions = np.array([body.position.copy() for body in self.bodies], dtype=float)
        velocities = np.array([body.velocity.copy() for body in self.bodies], dtype=float)
        accelerations = np.array([body.acceleration.copy() for body in self.bodies], dtype=float)
        previous_accelerations = np.array(
            [body.previous_acceleration.copy() for body in self.bodies],
            dtype=float,
        )
        return positions, velocities, accelerations, previous_accelerations

    def apply_state_arrays(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        previous_accelerations: np.ndarray | None = None,
    ) -> None:
        """
        Write state arrays back into the Body objects.
        """
        n_bodies = len(self.bodies)

        if positions.shape != (n_bodies, 2):
            raise ValueError(
                f"positions must have shape ({n_bodies}, 2), got {positions.shape}."
            )
        if velocities.shape != (n_bodies, 2):
            raise ValueError(
                f"velocities must have shape ({n_bodies}, 2), got {velocities.shape}."
            )
        if accelerations.shape != (n_bodies, 2):
            raise ValueError(
                f"accelerations must have shape ({n_bodies}, 2), got {accelerations.shape}."
            )
        if previous_accelerations is not None and previous_accelerations.shape != (n_bodies, 2):
            raise ValueError(
                f"previous_accelerations must have shape ({n_bodies}, 2), "
                f"got {previous_accelerations.shape}."
            )

        for i, body in enumerate(self.bodies):
            body.position = positions[i].copy()
            body.velocity = velocities[i].copy()
            body.acceleration = accelerations[i].copy()

            if previous_accelerations is not None:
                body.previous_acceleration = previous_accelerations[i].copy()

    def compute_accelerations(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute gravitational accelerations for all bodies from a supplied
        array of positions.
        """
        n_bodies = len(self.bodies)

        if positions.shape != (n_bodies, 2):
            raise ValueError(
                f"positions must have shape ({n_bodies}, 2), got {positions.shape}."
            )

        masses = np.array([body.mass for body in self.bodies], dtype=float)
        accelerations = np.zeros_like(positions, dtype=float)

        for i in range(n_bodies - 1):
            for j in range(i + 1, n_bodies):
                displacement = positions[j] - positions[i]
                distance = float(np.linalg.norm(displacement))

                if distance < EPSILON:
                    raise ZeroDivisionError(
                        f"Bodies '{self.bodies[i].name}' and '{self.bodies[j].name}' "
                        "became too close."
                    )

                factor = self.G / (distance**3)

                accelerations[i] += factor * masses[j] * displacement
                accelerations[j] -= factor * masses[i] * displacement

        return accelerations

    def update_current_accelerations(self) -> None:
        """Recompute and store the current accelerations for all bodies."""
        positions, _, _, _ = self.state_arrays()
        accelerations = self.compute_accelerations(positions)

        for body, acceleration in zip(self.bodies, accelerations):
            body.acceleration = acceleration.copy()

    def initialise_previous_accelerations(self) -> None:
        """
        Estimate a(t - dt) using a backward Taylor expansion of the positions.

        This is needed to start the Beeman integrator cleanly.
        """
        positions, velocities, accelerations, _ = self.state_arrays()
        previous_positions = positions - velocities * self.dt + 0.5 * accelerations * (self.dt**2)
        previous_accelerations = self.compute_accelerations(previous_positions)

        for body, previous_acceleration in zip(self.bodies, previous_accelerations):
            body.previous_acceleration = previous_acceleration.copy()

    def step(self) -> None:
        """Advance the system by one timestep using the selected integrator."""
        self.integrator.step(self, self.dt)
        self.time += self.dt
        self.step_count += 1

    def run(
        self,
        duration_years: float,
        record_every: int = 1,
        energy_log_every: int = 1,
        detect_periods: bool = True,
        period_targets: list[str] | None = None,
        stop_when_periods_complete: bool = False,
    ) -> SimulationResults:
        """
        Run the simulation forward for a chosen duration.

        Parameters
        ----------
        duration_years
            Total simulated time in years.
        record_every
            Store body positions every this many timesteps.
        energy_log_every
            Compute and store energies every this many timesteps.
        detect_periods
            If True, attempt to detect orbital periods during the run.
        period_targets
            Names of bodies whose periods should be detected. If omitted,
            all non-solar bodies are used.
        stop_when_periods_complete
            If True, stop the run early once all requested periods have
            been detected.

        Returns
        -------
        SimulationResults
            Structured container holding the recorded trajectories,
            energy history, detected periods, colours, and method name.
        """
        if duration_years <= 0.0:
            raise ValueError("duration_years must be positive.")
        if record_every < 1:
            raise ValueError("record_every must be at least 1.")
        if energy_log_every < 1:
            raise ValueError("energy_log_every must be at least 1.")

        total_steps = int(ceil(duration_years / self.dt))

        if period_targets is None:
            period_targets = [
                body.name.lower()
                for body in self.bodies
                if body.name.lower() != SUN_NAME.lower()
            ]
        else:
            period_targets = [name.lower() for name in period_targets]

        period_detector = None
        if detect_periods:
            period_detector = OrbitalPeriodDetector(period_targets, central_body_name=SUN_NAME)
            period_detector.initialise(self.bodies, self.time)

        times = [self.time]
        positions_history = {body.name.lower(): [body.position.copy()] for body in self.bodies}

        kinetic_0, potential_0, total_0 = total_energy_components(self.bodies, self.G)
        energy_times = [self.time]
        kinetic_energy = [kinetic_0]
        potential_energy = [potential_0]
        total_energy = [total_0]

        for step in range(1, total_steps + 1):
            self.step()

            if period_detector is not None:
                period_detector.update(self.bodies, self.time)

            if step % record_every == 0 or step == total_steps:
                times.append(self.time)
                for body in self.bodies:
                    positions_history[body.name.lower()].append(body.position.copy())

            if step % energy_log_every == 0 or step == total_steps:
                kinetic, potential, total = total_energy_components(self.bodies, self.G)
                energy_times.append(self.time)
                kinetic_energy.append(kinetic)
                potential_energy.append(potential)
                total_energy.append(total)

            if (
                period_detector is not None
                and stop_when_periods_complete
                and period_detector.all_found
            ):
                break

        positions_history_array = {
            name: np.vstack(history) for name, history in positions_history.items()
        }

        orbital_periods: dict[str, float] = {}
        if period_detector is not None:
            orbital_periods = dict(period_detector.periods)

        return SimulationResults(
            times=np.array(times, dtype=float),
            positions_history=positions_history_array,
            energy_times=np.array(energy_times, dtype=float),
            kinetic_energy=np.array(kinetic_energy, dtype=float),
            potential_energy=np.array(potential_energy, dtype=float),
            total_energy=np.array(total_energy, dtype=float),
            orbital_periods=orbital_periods,
            body_colours={body.name.lower(): body.colour for body in self.bodies},
            method_name=self.integrator.name,
        )