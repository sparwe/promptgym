from __future__ import annotations
from typing import Protocol, Dict, Any
from dataclasses import dataclass

@dataclass
class EvalAction:
    arm: int = 0
    task: int = 0
    meta: Dict[str, Any] | None = None   # strategy-specific metadata (e.g., phase, bucket)

@dataclass
class CreateAction:
    template_arm: int
    spec: Dict[str, Any] = None          # prompt spec; runner keeps it opaque

@dataclass
class StopAction:
    reason: str = ""

Action = EvalAction | CreateAction | StopAction

@dataclass
class EvalResult:
    arm: int
    task: int
    y: int
    meta: Optional[Dict[str, Any]] = None # may include env latency, etc.

@dataclass
class CreateResult:
    ok: bool
    arm_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

Result = EvalResult | CreateResult

class Strategy(Protocol):
    def name(self) -> str: ...
    def next_action(self) -> Action: ...
    def update(self, Result) -> None: ...
    def is_done(self) -> bool: ...
    def select_winner(self) -> int: ...
