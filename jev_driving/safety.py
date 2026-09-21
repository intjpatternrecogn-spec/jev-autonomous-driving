"""Transparent safety constraints for synthetic driving decisions."""

from __future__ import annotations

from .types import DrivingState


def normalise_action(action: str) -> str:
    return "_".join(action.lower().replace("-", " ").split())


def evaluate_action(state: DrivingState, action: str) -> tuple[bool, tuple[str, ...]]:
    name = normalise_action(action)
    reasons: list[str] = []

    closing_speed = max(0.0, -state.front_relative_speed_mps)
    ttc = state.front_distance_m / closing_speed if closing_speed > 0 else float("inf")
    immediate_stop = state.traffic_light.lower() == "red" or (
        state.pedestrian_distance_m is not None and state.pedestrian_distance_m < 12.0
    )

    if name == "change_left" and not state.left_lane_available:
        reasons.append("left lane is unavailable")
    if name == "change_right" and not state.right_lane_available:
        reasons.append("right lane is unavailable")
    if immediate_stop and name not in {"brake", "emergency_brake"}:
        reasons.append("stop constraint is active")
    if ttc < 2.0 and name in {"accelerate", "keep_lane"}:
        reasons.append("front time-to-collision is below 2 seconds")

    return not reasons, tuple(reasons)


def driving_prior(state: DrivingState, action: str) -> float:
    name = normalise_action(action)
    speed_error = state.speed_limit_mps - state.ego_speed_mps
    priors = {
        "keep_lane": 0.16,
        "change_left": -0.05,
        "change_right": -0.08,
        "brake": -0.10,
        "emergency_brake": -0.25,
        "accelerate": min(0.18, max(-0.18, speed_error / 30.0)),
    }
    if state.front_distance_m < 20.0 and name == "brake":
        return 0.28
    return priors.get(name, 0.0)
