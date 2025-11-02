from __future__ import annotations
from typing import Protocol, Dict, Any


@dataclass
class EvalAction:
    action: Literal["eval"] = "eval"
    arm: int = 0
    task: int = 0
    meta: Dict[str, Any] | None = None   # strategy-specific metadata (e.g., phase, bucket)

@dataclass
class CreateAction:
    action: Literal["create"] = "create"
    spec: Dict[str, Any] = None          # prompt spec; runner keeps it opaque

@dataclass
class StopAction:
    action: Literal["stop"] = "stop"
    reason: str = ""

Action = EvalAction | CreateAction | StopAction

@dataclass
class EvalResult:
    arm: int
    task: int
    y: int
    meta: Dict[str, Any] # may include env latency, etc.

@dataclass
class CreateResult:
    ok: bool
    arm_id: Optional[int] = None
    meta: Dict[str, Any] = None

Result = EvalResult | CreateResult

class Strategy(Protocol):
    def name(self) -> str: ...
    def next_action(self) -> Action: ...
    def update(self, Result) -> None: ...
    def is_done(self) -> bool: ...
    def select_winner(self) --> bool: ...
