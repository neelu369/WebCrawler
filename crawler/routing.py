from __future__ import annotations

from typing import Literal

from crawler.state import State


def route_after_evaluation(state: State) -> Literal["__end__", "intent_parser", "investigator"]:
    """Retry while gaps remain and retry budget is available."""
    if state.missing_data_targets:
        if state.retry_count <= state.max_retries:
            print(
                f"[Router] {len(state.missing_data_targets)} gaps found "
                f"(retry {state.retry_count}/{state.max_retries}) — triggering Investigator."
            )
            return "investigator"

        print(
            f"[Router] Retry budget exhausted "
            f"({state.retry_count}/{state.max_retries}) with remaining gaps — ending pipeline."
        )
    return "__end__"
