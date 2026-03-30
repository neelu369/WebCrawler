"""LLM wrapper passing calls directly to Replicate native client.

Uses the REPLICATE_API_TOKEN environment variable. 
This file acts as a small shim to maintain the existing pipeline usage:
`from crawler.llm import replicate`
`replicate.run(...)`
"""

from __future__ import annotations

import os
from typing import Any

# We use the native replicate python package which reads REPLICATE_API_TOKEN
import replicate as _replicate_native


class _ReplicateWrapper:
    """A wrapper proxy to intercept requests if needed and ensure compatibility."""
    
    def run(self, model: str, input: dict[str, Any], **kwargs: Any) -> list[str]:
        # Ensure token is present
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable is missing! "
                "Get a token at replicate.com and add it to your .env file."
            )
            
        try:
            # replicate.run returns a generator for text streams. 
            # The pipeline expects a list of strings we can "".join()
            output_iterator = _replicate_native.run(model, input=input, **kwargs)
            
            # Exhaust the generator to build the full response
            content = "".join([part for part in output_iterator])
            return [content]
        except Exception as exc:
            print(f"[LLM Error] Replicate request failed for model {model}: {exc}")
            raise exc


replicate = _ReplicateWrapper()
