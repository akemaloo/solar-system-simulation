from __future__ import annotations

import math

SUN_NAME = "sun"

# Units used throughout the project:
# - distance: AU
# - mass: Earth masses
# - time: years
#
# With these choices, for a 1 AU circular orbit around the Sun:
# GM_sun = 4*pi^2
# Since M_sun = 332946 Earth masses, G is:
G = 4.0 * math.pi**2 / 332_946.0

DEFAULT_DT = 1.0 / 2000.0
DEFAULT_SIMULATION_YEARS = 13.0
EPSILON = 1.0e-12

# Sidereal orbital periods in Earth years
ACTUAL_ORBITAL_PERIODS = {
    "mercury": 0.2408467,
    "venus": 0.61519726,
    "earth": 1.0,
    "mars": 1.8808158,
    "jupiter": 11.862615,
}