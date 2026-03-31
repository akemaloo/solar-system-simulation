from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Body:
    """A single gravitating body in the simulation."""

    name: str
    mass: float
    colour: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    previous_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.acceleration = np.asarray(self.acceleration, dtype=float)
        self.previous_acceleration = np.asarray(self.previous_acceleration, dtype=float)

    def kinetic_energy(self) -> float:
        """Return the kinetic energy of this body."""
        return 0.5 * self.mass * float(np.dot(self.velocity, self.velocity))