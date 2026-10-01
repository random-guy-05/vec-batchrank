"""Published Virtual Embryo Challenge metric definitions and validation anchors.

Snapshot date: 2026-09-30.
Source: https://virtualembryo.ai/challenge/evaluation

Only metrics that currently contribute to the official task score are listed. veckit may
return additional diagnostics; BatchRank preserves them in raw outputs but does not silently
fold them into the official score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower", "zero"]


@dataclass(frozen=True)
class MetricSpec:
    direction: Direction
    weight: float


@dataclass(frozen=True)
class BoardSpec:
    task: str
    anchors: dict[str, tuple[float, float]]  # metric -> (floor, ceiling)


TASK_SPECS: dict[str, dict[str, MetricSpec]] = {
    "T1": {
        "de_score": MetricSpec("higher", 0.25),
        "de_direction": MetricSpec("higher", 0.25),
        "mmd_u": MetricSpec("lower", 0.30),
        "variogram": MetricSpec("lower", 0.20),
    },
    "T2": {
        "de_score": MetricSpec("higher", 0.25 * 0.50),
        "de_direction": MetricSpec("higher", 0.25 * 0.50),
        "mmd_u": MetricSpec("lower", 0.25 * 0.60),
        "variogram": MetricSpec("lower", 0.25 * 0.40),
        "d2_shape": MetricSpec("lower", 0.25 / 3.0),
        "occupancy_dice": MetricSpec("higher", 0.25 / 3.0),
        "scale_log_ratio": MetricSpec("zero", 0.25 / 3.0),
        "neighborhood_mmd": MetricSpec("lower", 0.25),
    },
    "T3": {
        "de_score": MetricSpec("higher", 0.30),
        "de_direction": MetricSpec("higher", 0.25),
        "severity_slope": MetricSpec("zero", 0.25),
        "mmd_u": MetricSpec("lower", 0.20 * 0.60),
        "variogram": MetricSpec("lower", 0.20 * 0.40),
    },
}

# Published validation-board anchors. Hidden test anchors are intentionally unavailable.
BOARD_SPECS: dict[str, BoardSpec] = {
    "T1:val": BoardSpec(
        "T1",
        {
            "de_score": (0.0, 0.8464),
            "de_direction": (0.0, 0.7901),
            "mmd_u": (0.08359, 0.00406),
            "variogram": (0.005219, 0.000158),
        },
    ),
    "T2:embryo:val_interp": BoardSpec(
        "T2",
        {
            "de_score": (0.0, 0.8413),
            "de_direction": (0.0, 0.9185),
            "mmd_u": (0.08571, 0.00307),
            "variogram": (0.053533, 0.001264),
            "d2_shape": (0.05306, 0.00268),
            "occupancy_dice": (0.7047, 0.7661),
            "scale_log_ratio": (-0.3063, 0.0053),
            "neighborhood_mmd": (0.21105, 0.01128),
        },
    ),
    "T2:heart:val_extrap": BoardSpec(
        "T2",
        {
            "de_score": (0.0, 0.9420),
            "de_direction": (0.0, 0.9915),
            "mmd_u": (0.02455, 0.00011),
            "variogram": (0.031712, 0.000696),
            "d2_shape": (0.00933, 0.00342),
            "occupancy_dice": (0.7747, 0.9465),
            "scale_log_ratio": (-0.2680, 0.0074),
            "neighborhood_mmd": (0.07184, 0.0004),
        },
    ),
    "T2:heart:val_interp": BoardSpec(
        "T2",
        {
            "de_score": (0.0, 0.8182),
            "de_direction": (0.0, 0.9553),
            "mmd_u": (0.0207, 0.00087),
            "variogram": (0.023828, 0.001021),
            "d2_shape": (0.03079, 0.00412),
            "occupancy_dice": (0.6748, 0.8796),
            "scale_log_ratio": (0.3284, -0.0119),
            "neighborhood_mmd": (0.05728, 0.00432),
        },
    ),
    "T3:gata4": BoardSpec(
        "T3",
        {
            "de_score": (0.0, 0.8696),
            "de_direction": (0.0, 0.9247),
            "severity_slope": (-6.9078, -0.0088),
            "mmd_u": (0.03141, 0.00039),
            "variogram": (0.032212, 0.000737),
        },
    ),
}
