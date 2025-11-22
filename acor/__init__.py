"""
ACOR project package.

Implements Adaptive Collaborative Organization and Robustness framework
with dynamic fuzzy grouping, trust assessment, and hierarchical consensus.
"""

from .models.acor_policy import ACORPolicy
from .runners.acor_trainer import ACORTrainer
from .runners.mappo_trainer import MAPPOTrainer

__all__ = ["ACORPolicy", "ACORTrainer", "MAPPOTrainer"]
