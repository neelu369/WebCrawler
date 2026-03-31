from crawler.nodes.preprocessor import (
    _coerce_metrics,
    _extract_incubator_entities_fallback,
    _try_parse_entities_payload,
)


def test_try_parse_entities_payload_handles_fenced_json_with_prose() -> None:
    raw = """
Here is the extracted output:
```json
[
  {"name": "NSRCEL Incubator", "description": "IIM Bangalore program", "metrics": {"Location": "Bengaluru"}, "priority_score": 0.9}
]
```
Thanks.
"""
    entities = _try_parse_entities_payload(raw)
    assert len(entities) == 1
    assert entities[0]["name"] == "NSRCEL Incubator"


def test_coerce_metrics_accepts_list_payload() -> None:
    metrics = _coerce_metrics(["Govt backed", "Academic partner"])
    assert metrics["Metric 1"] == "Govt backed"
    assert metrics["Metric 2"] == "Academic partner"


def test_fallback_extracts_incubator_entities() -> None:
    text = (
        "NSRCEL Incubator is one of the most active startup programs. "
        "CIIE.CO Accelerator supports early-stage founders."
    )
    entities = _extract_incubator_entities_fallback(text, "startup incubators in India")
    names = {e["name"] for e in entities}
    assert "NSRCEL Incubator" in names
    assert "CIIE.CO Accelerator" in names
