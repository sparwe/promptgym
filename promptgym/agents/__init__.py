from .base import BaseAgent
from .random_search import RandomAgent
from .first_arm import FirstArm

__all__ = ["BaseAgent", "RandomAgent", "FirstArm", "REGISTRY"]

REGISTRY = {
    "random": RandomAgent,
    "first": FirstArm,
}

