"""LLM cost tracker for pipeline usage.

Uses contextvars to isolate costs per async task, so concurrent pipeline
jobs don't mix their cost data.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field


# ── Pricing (USD per token) ──────────────────────────────────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3-70b-instruct": {
        "input": 0.65 / 1_000_000,
        "output": 2.75 / 1_000_000,
    },
    "meta-llama/llama-3-8b-instruct": {
        "input": 0.05 / 1_000_000,
        "output": 0.25 / 1_000_000,
    },
    # Legacy Replicate-format keys (kept for backward compatibility)
    "meta/meta-llama-3-70b-instruct": {
        "input": 0.65 / 1_000_000,
        "output": 2.75 / 1_000_000,
    },
    "meta/meta-llama-3-8b-instruct": {
        "input": 0.05 / 1_000_000,
        "output": 0.25 / 1_000_000,
    },
}

_DEFAULT_PRICING = {"input": 0.10 / 1_000_000, "output": 0.50 / 1_000_000}


@dataclass
class LLMCall:
    """A single recorded LLM invocation."""

    node: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Accumulates LLM usage across all nodes in a single pipeline run."""

    def __init__(self) -> None:
        self._calls: list[LLMCall] = []

    def record(
        self,
        *,
        node: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float = 0.0,
    ) -> LLMCall:
        """Record one LLM call and return the entry."""
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        entry = LLMCall(
            node=node,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_s=latency_s,
        )
        self._calls.append(entry)
        return entry

    def get_summary(self) -> dict:
        """Return a cost breakdown by node and a grand total."""
        by_node: dict[str, dict] = {}
        for c in self._calls:
            entry = by_node.setdefault(
                c.node,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_s": 0.0,
                },
            )
            entry["calls"] += 1
            entry["input_tokens"] += c.input_tokens
            entry["output_tokens"] += c.output_tokens
            entry["cost_usd"] += c.cost_usd
            entry["latency_s"] += c.latency_s

        total_cost = sum(e["cost_usd"] for e in by_node.values())
        total_tokens = sum(
            e["input_tokens"] + e["output_tokens"] for e in by_node.values()
        )

        return {
            "by_node": by_node,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "total_calls": len(self._calls),
        }

    def print_report(self) -> None:
        """Pretty-print a cost report to stdout."""
        summary = self.get_summary()
        print("\n+------------------- COST REPORT -------------------+")
        for node, data in summary["by_node"].items():
            print(
                f"|  {node:<20s}  "
                f"calls={data['calls']}  "
                f"tokens={data['input_tokens'] + data['output_tokens']:>6}  "
                f"cost=${data['cost_usd']:.6f}"
            )
        print("+---------------------------------------------------+")
        print(
            f"|  TOTAL               "
            f"calls={summary['total_calls']}  "
            f"tokens={summary['total_tokens']:>6}  "
            f"cost=${summary['total_cost_usd']:.6f}"
        )
        print("+---------------------------------------------------+\n")


# ── Per-task isolation via contextvars ────────────────────────────────
# asyncio.create_task() copies the current context, so each pipeline job
# gets its own CostTracker instance when new_tracker() is called at the
# start of the task.

_tracker_var: ContextVar[CostTracker] = ContextVar("cost_tracker")


def new_tracker() -> CostTracker:
    """Create and set a fresh CostTracker for the current async context."""
    t = CostTracker()
    _tracker_var.set(t)
    return t


def get_tracker() -> CostTracker:
    """Return the CostTracker for the current async context."""
    try:
        return _tracker_var.get()
    except LookupError:
        return new_tracker()


class _TrackerProxy:
    """Module-level proxy that delegates to the context-local CostTracker.

    All existing ``from crawler.cost_tracker import tracker`` imports
    continue to work unchanged — calls are forwarded to the task-local
    tracker via the ContextVar.
    """

    def record(self, **kwargs):  # type: ignore[override]
        return get_tracker().record(**kwargs)

    def get_summary(self):
        return get_tracker().get_summary()

    def print_report(self):
        return get_tracker().print_report()


# Backward-compatible module-level singleton — now a thin proxy.
tracker = _TrackerProxy()
