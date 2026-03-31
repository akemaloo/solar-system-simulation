from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulation import Simulation


class Integrator(ABC):
    """Base class for time integration methods."""

    name = "Integrator"

    @abstractmethod
    def step(self, simulation: "Simulation", dt: float) -> None:
        """Advance the simulation by one timestep."""
        raise NotImplementedError


class BeemanIntegrator(Integrator):
    """Beeman predictor-corrector style integrator."""

    name = "Beeman"

    def step(self, simulation: "Simulation", dt: float) -> None:
        positions, velocities, accelerations, previous_accelerations = simulation.state_arrays()

        positions_new = (
            positions
            + velocities * dt
            + ((4.0 * accelerations - previous_accelerations) * (dt**2) / 6.0)
        )

        accelerations_new = simulation.compute_accelerations(positions_new)

        velocities_new = (
            velocities
            + ((2.0 * accelerations_new + 5.0 * accelerations - previous_accelerations) * dt / 6.0)
        )

        simulation.apply_state_arrays(
            positions_new,
            velocities_new,
            accelerations_new,
            accelerations,
        )


class EulerCromerIntegrator(Integrator):
    """Euler-Cromer integrator."""

    name = "Euler-Cromer"

    def step(self, simulation: "Simulation", dt: float) -> None:
        positions, velocities, accelerations, _ = simulation.state_arrays()

        velocities_new = velocities + accelerations * dt
        positions_new = positions + velocities_new * dt
        accelerations_new = simulation.compute_accelerations(positions_new)

        simulation.apply_state_arrays(
            positions_new,
            velocities_new,
            accelerations_new,
            accelerations,
        )


class DirectEulerIntegrator(Integrator):
    """Direct (forward) Euler integrator."""

    name = "Direct Euler"

    def step(self, simulation: "Simulation", dt: float) -> None:
        positions, velocities, accelerations, _ = simulation.state_arrays()

        positions_new = positions + velocities * dt
        velocities_new = velocities + accelerations * dt
        accelerations_new = simulation.compute_accelerations(positions_new)

        simulation.apply_state_arrays(
            positions_new,
            velocities_new,
            accelerations_new,
            accelerations,
        )