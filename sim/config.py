"""Simulation settings. Edit config.json to change them without touching code.

Scale: 1 px = 55.7 mm. A 10 x 10 px body has a 7.07 px circumscribed radius, which matches the
787 mm rotating diameter of the reference AMR. Speeds and accelerations below are that AMR's
datasheet values converted to px.
"""
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    map: str = "cmpp_shelf"               # maps/<map>.json
    num_amrs: int = 20
    seed: int = 0
    dt: float = 0.1                       # time step [s]

    amr_length: float = 10.0              # body size [px]
    amr_width: float = 10.0
    max_speed: float = 14.37              # [px/s]    0.8 m/s operating speed
    accel: float = 5.39                   # [px/s^2]  0.3 m/s^2, also used for braking
    max_angular_velocity: float = 0.7854  # [rad/s]   45 deg/s
    max_angular_accel: float = 0.7854     # [rad/s^2]

    @property
    def map_path(self) -> Path:
        return ROOT / "maps" / f"{self.map}.json"

    @classmethod
    def load(cls, path=ROOT / "config.json") -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            print(f"[config] ignoring unknown keys in {path.name}: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in names})
