"""Metrics Evaluator node — checks structured results against target metrics."""
from __future__ import annotations
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from crawler.state import State

async def evaluate_metrics(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    max_retries = state.max_retries
    try:
        cfg = (config or {}).get("configurable", {})
        if "max_retries" in cfg:
            max_retries = int(cfg["max_retries"])
    except Exception:
        pass

    if not state.target_metrics:
        print("[MetricsEvaluator] No target metrics. Skipping.")
        return {"missing_data_targets": [], "max_retries": max_retries}
    if not state.structured_results:
        print("[MetricsEvaluator] No structured results. Skipping.")
        return {"missing_data_targets": [], "max_retries": max_retries}

    missing_targets = []
    for entity in state.structured_results:
        for metric in state.target_metrics:
            metric_lower = metric.lower()
            found = any(metric_lower in k.lower() for k in entity.properties.keys())
            if not found:
                found = any(metric_lower in str(r.get("type","")).lower() or metric_lower in str(r.get("predicate","")).lower() for r in entity.relationships)
            if not found and entity.description:
                found = metric_lower in entity.description.lower()
            if not found:
                missing_targets.append(f"{entity.name} :: {metric}")

    unique = list(dict.fromkeys(missing_targets))[:15]
    if unique:
        print(f"[MetricsEvaluator] {len(unique)} missing targets")
        for t in unique: print(f"  -> {t}")
        # retry_count uses Annotated[int, operator.add] in State, so returning 1
        # means "add 1 to the current count" — NOT "set count to 1".
        return {
            "missing_data_targets": unique,
            "retry_count": 1,
            "max_retries": max_retries,
        }
    else:
        print(f"[MetricsEvaluator] All {len(state.target_metrics)} metrics satisfied")
    return {"missing_data_targets": unique, "max_retries": max_retries}
