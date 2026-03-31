from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .body import Body
from .constants import G

if TYPE_CHECKING:
    from .simulation import SimulationResults


def load_bodies_from_json(json_path: str | Path) -> list[Body]:
    """
    Load bodies from JSON and initialise:
    - planets on the positive x-axis
    - circular-orbit approximation for initial velocity
    - Sun velocity chosen to make the total momentum zero
    - positions shifted so the initial centre of mass is at the origin
    """
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    raw_bodies = data.get("bodies", [])
    if not raw_bodies:
        raise ValueError("No bodies found in JSON file.")

    sun_record = None
    for record in raw_bodies:
        if record["name"].lower() == "sun":
            sun_record = record
            break

    if sun_record is None:
        raise ValueError("The JSON file must contain a body named 'sun'.")

    sun = Body(
        name=sun_record["name"].lower(),
        mass=float(sun_record["mass"]),
        colour=sun_record.get("colour", "y"),
        position=np.zeros(2, dtype=float),
        velocity=np.zeros(2, dtype=float),
    )

    bodies = [sun]

    for record in raw_bodies:
        name = record["name"].lower()
        if name == "sun":
            continue

        orbital_radius = float(record["orbital_radius"])
        speed = math.sqrt(G * sun.mass / orbital_radius)

        body = Body(
            name=name,
            mass=float(record["mass"]),
            colour=record.get("colour", "white"),
            position=np.array([orbital_radius, 0.0], dtype=float),
            velocity=np.array([0.0, speed], dtype=float),
        )
        bodies.append(body)

    total_planet_momentum = np.sum(
        [body.mass * body.velocity for body in bodies[1:]],
        axis=0,
    )
    sun.velocity = -total_planet_momentum / sun.mass

    total_mass = sum(body.mass for body in bodies)
    centre_of_mass = np.sum(
        [body.mass * body.position for body in bodies],
        axis=0,
    ) / total_mass

    for body in bodies:
        body.position = body.position - centre_of_mass

    return bodies


def save_energy_csv(results: "SimulationResults", output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_years", "kinetic_energy", "potential_energy", "total_energy"])

        for time, kinetic, potential, total in zip(
            results.energy_times,
            results.kinetic_energy,
            results.potential_energy,
            results.total_energy,
        ):
            writer.writerow([time, kinetic, potential, total])


def save_period_comparison_csv(
    simulated_periods: dict[str, float],
    actual_periods: dict[str, float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["planet", "actual_years", "simulated_years", "percent_error"])

        for name, actual in actual_periods.items():
            simulated = simulated_periods.get(name)
            if simulated is None:
                writer.writerow([name, actual, "", ""])
            else:
                percent_error = 100.0 * (simulated - actual) / actual
                writer.writerow([name, actual, simulated, percent_error])