from .base import BaseAgent
from .random_search import RandomAgent
from .first_arm import FirstArm

REGISTRY = {
    "random": RandomAgent,
    "first": FirstArm,
}

