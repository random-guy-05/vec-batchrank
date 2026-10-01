"""VEC BatchRank."""

__version__ = "1.0.0"

from .ranking import BOARD_SPECS, TASK_SPECS, official_score, pareto_front

__all__ = ["BOARD_SPECS", "TASK_SPECS", "official_score", "pareto_front"]
