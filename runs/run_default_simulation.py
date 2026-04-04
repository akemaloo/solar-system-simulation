from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_sim import (
    BeemanIntegrator,
    DEFAULT_DT,
    DEFAULT_SIMULATION_YEARS,
    Simulation,
    load_bodies_from_json,
    save_energy_csv,
)
from solar_sim.plotting import (
    animate_orbits,
    plot_energy_history,
    plot_orbits,
    plot_total_energy_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "parameters_solar.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLANET_ORDER = ["mercury", "venus", "earth", "mars", "jupiter"]


def print_detected_periods(periods: dict[str, float]) -> None:
    """Print detected orbital periods in a clean table."""
    print("\nDetected orbital periods from default simulation")
    print("-" * 46)
    print(f"{'Planet':<12}{'Period [years]':>18}")
    print("-" * 46)

    for name in PLANET_ORDER:
        period = periods.get(name)
        if period is None:
            print(f"{name.capitalize():<12}{'not detected':>18}")
        else:
            print(f"{name.capitalize():<12}{period:>18.6f}")

    print("-" * 46)


def main() -> None:
    bodies = load_bodies_from_json(DATA_FILE)

    simulation = Simulation(
        bodies=bodies,
        integrator=BeemanIntegrator(),
        dt=DEFAULT_DT,
    )

    results = simulation.run(
        duration_years=DEFAULT_SIMULATION_YEARS,
        record_every=20,
        energy_log_every=20,
        detect_periods=True,
        stop_when_periods_complete=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_energy_csv(results, OUTPUT_DIR / "default_energy.csv")

    energy_figure, _ = plot_energy_history(
        results,
        title="Default simulation: energy history",
        save_path=OUTPUT_DIR / "default_energy.png",
    )
    energy_error_figure, _ = plot_total_energy_error(
        results,
        title="Default simulation: relative total-energy drift",
        save_path=OUTPUT_DIR / "default_energy_error.png",
    )
    static_orbit_figure, _ = plot_orbits(
        results,
        title="Default simulation: orbital paths",
        save_path=OUTPUT_DIR / "default_orbits.png",
    )

    print_detected_periods(results.orbital_periods)
    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print("Close the animation window to finish the script.")

    is_interactive_backend = (
        getattr(static_orbit_figure.canvas, "required_interactive_framework", None) is not None
    )

    plt.close(energy_figure)
    plt.close(energy_error_figure)
    plt.close(static_orbit_figure)

    if not is_interactive_backend:
        print("Non-interactive matplotlib backend detected, so the animation window was not shown.")
        return

    orbit_figure, orbit_animation = animate_orbits(
        results,
        title="Default simulation: animated solar system (3D view)",
        interval=20,
        trail_length=150,
        frame_stride=1,
    )

    # Keep a reference alive so matplotlib does not garbage-collect the animation.
    orbit_figure._solar_animation = orbit_animation
    plt.show()


if __name__ == "__main__":
    main()