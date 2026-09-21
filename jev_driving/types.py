"""Public types for the driving decision engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class DrivingState:
    ego_speed_mps: float
    speed_limit_mps: float
    front_distance_m: float
    front_relative_speed_mps: float
    left_lane_available: bool
    right_lane_available: bool
    pedestrian_distance_m: float | None
    traffic_light: str

    def __post_init__(self) -> None:
        if self.ego_speed_mps < 0 or self.speed_limit_mps < 0:
            raise ValueError("speeds must be non-negative")
        if self.front_distance_m < 0:
            raise ValueError("front distance must be non-negative")
        if self.pedestrian_distance_m is not None and self.pedestrian_distance_m < 0:
            raise ValueError("pedestrian distance must be non-negative")
        if self.traffic_light.lower() not in {"red", "yellow", "green", "unknown"}:
            raise ValueError("traffic_light must be red, yellow, green or unknown")

    def to_text(self) -> str:
        fields = asdict(self)
        return "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in fields.items())


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    probabilities: Mapping[str, float]
    feasible: Mapping[str, bool]
    reasons: Mapping[str, tuple[str, ...]]
    confidence: float
    safe: bool
