from .base import BaseAgent
from .random_search import RandomAgent
from .first_arm import FirstArm
from .ucb_sampler import UCBSampler

__all__ = ["BaseAgent", "RandomAgent", "FirstArm", "UCBSampler", "REGISTRY"]

REGISTRY = {
    "random": RandomAgent,
    "first": FirstArm,
    "ucb": UCBSampler,
}

