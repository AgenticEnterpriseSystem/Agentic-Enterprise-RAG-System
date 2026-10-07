"""Remaining per-operation allowance for cooperating model providers.

Context propagation preserves the simple complete(system, prompt) protocol.
Providers must opt in; synchronous retrievers and other providers remain soft
limits checked between operations and cannot be forcibly interrupted.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time

from src.contracts import Usage


@dataclass
class CallBudget:
    remaining_tokens: int
    remaining_seconds: float
    remaining_cost_usd: float
    deadline: float = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        self.deadline = time.monotonic() + self.remaining_seconds

    def seconds_left(self):
        """Include time spent in retry delays or code between provider calls."""
        return max(0, min(self.remaining_seconds, self.deadline - time.monotonic()))

    def consume(self, *, tokens=0, seconds=0, cost_usd=0):
        """Debit completed attempts, including retries within one operation."""
        self.remaining_tokens = max(0, self.remaining_tokens - tokens)
        self.remaining_seconds = max(0, self.remaining_seconds - seconds)
        self.remaining_cost_usd = max(0, self.remaining_cost_usd - cost_usd)


class RunBudgetExceeded(Exception):
    """A provider rejected a call before exceeding its remaining allowance."""

    def __init__(self, message="Model call exceeds remaining run budget.", usage=None):
        super().__init__(message)
        self.usage = usage or Usage(estimated_cost_usd=0)


_active_budget = ContextVar("agent_call_budget", default=None)


def current_call_budget():
    return _active_budget.get()


@contextmanager
def call_budget(budget):
    token = _active_budget.set(budget)
    try:
        yield budget
    finally:
        _active_budget.reset(token)
