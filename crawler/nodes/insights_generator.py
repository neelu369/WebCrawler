"""Insights Generator node — builds explainability findings from pipeline outputs.

Uses structured Neo4j results plus raw verified source metadata to generate:
1. A user-facing summary string
2. A list of citation-backed insight findings with confidence scores

The node is deterministic by default. Optional LLM synthesis can be enabled via
Configuration.enable_insights_llm_synthesis for summary polishing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from crawler.llm import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import StructuredResult
from crawler.state import State

_MISSING_VALUES = frozenset(
	{
		"",
		"n/a",
		"na",
		"null",
		"none",
		"not available",
		"not found",
		"unknown",
		"-",
		"–",
		"?",
	}
)


def _is_missing(value: Any) -> bool:
	if value is None:
		return True
	return str(value).strip().lower() in _MISSING_VALUES


def _extract_number(value: Any) -> float | None:
	if _is_missing(value):
		return None

	text = str(value).strip()
	cleaned = re.sub(r"[,$%₹€£¥~\s]", "", text)
	lower = cleaned.lower()

	multipliers = [
		("bn", 1_000_000_000),
		("b", 1_000_000_000),
		("mn", 1_000_000),
		("m", 1_000_000),
		("lakh", 100_000),
		("lac", 100_000),
		("cr", 10_000_000),
		("k", 1_000),
	]
	for suffix, mult in multipliers:
		if lower.endswith(suffix):
			try:
				return float(lower[: -len(suffix)]) * mult
			except ValueError:
				break

	try:
		return float(cleaned)
	except ValueError:
		pass

	m = re.search(r"[-+]?\d+\.?\d*", text)
	if m:
		try:
			return float(m.group())
		except ValueError:
			return None
	return None


def _source_lookup(state: State) -> dict[str, dict[str, Any]]:
	sources: dict[str, dict[str, Any]] = {}
	for src in state.verified_sources:
		if src.url:
			sources[src.url] = {
				"source_url": src.url,
				"source_trust": float(src.credibility_score),
				"is_trusted": bool(src.is_trusted),
				"relevance_score": float(src.relevance_score),
				"content": src.content,
			}
	return sources


def _content_snippet(content: str, needle: str, max_len: int = 180) -> str:
	if not content:
		return ""
	text = re.sub(r"\s+", " ", content).strip()
	if not needle:
		return text[:max_len]
	idx = text.lower().find(str(needle).lower())
	if idx == -1:
		return text[:max_len]
	start = max(0, idx - 40)
	end = min(len(text), idx + max_len - 40)
	return text[start:end]


def _confidence_from_evidence(evidence: list[dict[str, Any]], *, consistency: float = 0.85) -> float:
	if not evidence:
		return 0.2
	trust_scores = []
	for ev in evidence:
		trust = float(ev.get("source_trust", 0.5))
		if ev.get("is_trusted"):
			trust = min(1.0, trust + 0.15)
		trust_scores.append(trust)

	avg_trust = sum(trust_scores) / len(trust_scores)
	coverage = min(1.0, len(evidence) / 3.0)
	score = (0.45 * avg_trust) + (0.35 * coverage) + (0.20 * max(0.0, min(1.0, consistency)))
	return round(max(0.0, min(1.0, score)), 3)


def _metric_values(results: list[StructuredResult]) -> dict[str, list[dict[str, Any]]]:
	by_metric: dict[str, list[dict[str, Any]]] = {}
	for entity in results:
		for metric, value in entity.properties.items():
			by_metric.setdefault(metric, []).append(
				{
					"entity": entity,
					"value": value,
					"numeric": _extract_number(value),
				}
			)
	return by_metric


def _metric_coverage_findings(state: State, source_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
	if not state.target_metrics or not state.structured_results:
		return []

	findings: list[dict[str, Any]] = []
	for metric in state.target_metrics:
		present = 0
		missing_entities: list[str] = []
		evidence: list[dict[str, Any]] = []

		for entity in state.structured_results:
			keys = [k for k in entity.properties.keys() if metric.lower() in k.lower()]
			if keys:
				present += 1
				metric_key = keys[0]
				source_url = entity.source_urls[0] if entity.source_urls else ""
				src_meta = source_map.get(source_url, {})
				evidence.append(
					{
						"entity": entity.name,
						"metric": metric_key,
						"value": str(entity.properties.get(metric_key, "")),
						"source_url": source_url,
						"source_trust": src_meta.get("source_trust", 0.5),
						"is_trusted": src_meta.get("is_trusted", False),
						"evidence_snippet": _content_snippet(src_meta.get("content", ""), str(entity.properties.get(metric_key, ""))),
					}
				)
			else:
				missing_entities.append(entity.name)

		total = len(state.structured_results)
		if total == 0:
			continue

		findings.append(
			{
				"title": f"Coverage for {metric}",
				"finding": f"{present}/{total} entities include '{metric}' values.",
				"type": "coverage" if not missing_entities else "caution",
				"confidence": _confidence_from_evidence(evidence, consistency=0.8),
				"entities_involved": [e.name for e in state.structured_results],
				"evidence": evidence,
				"caveats": [
					f"Missing for: {', '.join(missing_entities)}" if missing_entities else ""
				],
			}
		)

	for item in findings:
		item["caveats"] = [c for c in item["caveats"] if c]
	return findings


def _comparison_findings(results: list[StructuredResult], source_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
	findings: list[dict[str, Any]] = []
	metrics = _metric_values(results)

	for metric, rows in metrics.items():
		numeric_rows = [r for r in rows if r["numeric"] is not None]
		if len(numeric_rows) < 2:
			continue

		sorted_rows = sorted(numeric_rows, key=lambda r: r["numeric"])
		low = sorted_rows[0]
		high = sorted_rows[-1]

		high_entity: StructuredResult = high["entity"]
		low_entity: StructuredResult = low["entity"]

		high_url = high_entity.source_urls[0] if high_entity.source_urls else ""
		low_url = low_entity.source_urls[0] if low_entity.source_urls else ""
		high_src = source_map.get(high_url, {})
		low_src = source_map.get(low_url, {})

		evidence = [
			{
				"entity": high_entity.name,
				"metric": metric,
				"value": str(high["value"]),
				"source_url": high_url,
				"source_trust": high_src.get("source_trust", 0.5),
				"is_trusted": high_src.get("is_trusted", False),
				"evidence_snippet": _content_snippet(high_src.get("content", ""), str(high["value"])),
			},
			{
				"entity": low_entity.name,
				"metric": metric,
				"value": str(low["value"]),
				"source_url": low_url,
				"source_trust": low_src.get("source_trust", 0.5),
				"is_trusted": low_src.get("is_trusted", False),
				"evidence_snippet": _content_snippet(low_src.get("content", ""), str(low["value"])),
			},
		]

		spread = abs(float(high["numeric"]) - float(low["numeric"]))
		findings.append(
			{
				"title": f"Spread in {metric}",
				"finding": (
					f"{high_entity.name} leads on {metric} ({high['value']}), "
					f"while {low_entity.name} is lowest ({low['value']})."
				),
				"type": "comparison",
				"confidence": _confidence_from_evidence(evidence, consistency=0.9),
				"entities_involved": [high_entity.name, low_entity.name],
				"evidence": evidence,
				"caveats": [f"Numeric spread: {spread:.3f}"],
			}
		)

	return findings


def _entity_profile_findings(results: list[StructuredResult], source_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
	findings: list[dict[str, Any]] = []
	for entity in results:
		populated = [(k, v) for k, v in entity.properties.items() if not _is_missing(v)]
		source_url = entity.source_urls[0] if entity.source_urls else ""
		src_meta = source_map.get(source_url, {})

		evidence = [
			{
				"entity": entity.name,
				"metric": k,
				"value": str(v),
				"source_url": source_url,
				"source_trust": src_meta.get("source_trust", 0.5),
				"is_trusted": src_meta.get("is_trusted", False),
				"evidence_snippet": _content_snippet(src_meta.get("content", ""), str(v)),
			}
			for k, v in populated
		]

		findings.append(
			{
				"title": f"Entity profile: {entity.name}",
				"finding": (
					f"{entity.name} includes {len(populated)} captured metric values"
					f" and {len(entity.relationships)} relationship facts."
				),
				"type": "trend",
				"confidence": _confidence_from_evidence(evidence, consistency=0.75),
				"entities_involved": [entity.name],
				"evidence": evidence,
				"caveats": [],
			}
		)
	return findings


def _build_default_summary(*, user_query: str, entities: int, findings: int, sources: int) -> str:
	return (
		f"For query '{user_query}', generated {findings} explainability insights from "
		f"{entities} structured entities backed by {sources} verified web sources."
	)


def _synthesise_summary_with_llm(
	*,
	configuration: Configuration,
	user_query: str,
	findings: list[dict[str, Any]],
) -> str | None:
	if not findings:
		return None

	sample = findings[:50]
	prompt = (
		"You are an explainability writer for a ranking system. "
		"Given these JSON findings, write a concise user-facing summary in 2-4 sentences. "
		"Mention key strengths and any important caveats. Return plain text only.\n\n"
		f"User query: {user_query}\n"
		f"Findings JSON sample: {json.dumps(sample, ensure_ascii=True)}"
	)

	t0 = time.time()
	try:
		output = replicate.run(
			configuration.model,
			input={"prompt": prompt, "max_tokens": 256, "temperature": 0.2},
		)
		raw = "".join(str(c) for c in output).strip()
		tracker.record(
			node="insights_generator",
			model=configuration.model,
			input_tokens=len(prompt) // 4,
			output_tokens=len(raw) // 4,
			latency_s=time.time() - t0,
		)
		return raw if raw else None
	except Exception as exc:
		print(f"[InsightsGenerator] LLM synthesis failed: {exc}")
		return None


async def generate_insights(
	state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
	configuration = Configuration.from_runnable_config(config)

	if not configuration.enable_insights_node:
		return {
			"insights_summary": "",
			"insights_items": [],
			"insights_metadata": {"enabled": False, "reason": "disabled_by_config"},
		}

	if not state.structured_results:
		return {
			"insights_summary": "No structured results available for insights generation.",
			"insights_items": [],
			"insights_metadata": {
				"enabled": True,
				"reason": "no_structured_results",
				"source_count": len(state.verified_sources),
			},
		}

	source_map = _source_lookup(state)

	findings: list[dict[str, Any]] = []
	findings.extend(_comparison_findings(state.structured_results, source_map))
	findings.extend(_metric_coverage_findings(state, source_map))
	findings.extend(_entity_profile_findings(state.structured_results, source_map))

	findings.sort(key=lambda f: float(f.get("confidence", 0.0)), reverse=True)

	summary = _build_default_summary(
		user_query=state.user_query,
		entities=len(state.structured_results),
		findings=len(findings),
		sources=len(source_map),
	)

	if configuration.enable_insights_llm_synthesis:
		llm_summary = _synthesise_summary_with_llm(
			configuration=configuration,
			user_query=state.user_query,
			findings=findings,
		)
		if llm_summary:
			summary = llm_summary

	metadata: dict[str, Any] = {
		"enabled": True,
		"source_count": len(source_map),
		"structured_entity_count": len(state.structured_results),
		"total_findings": len(findings),
		"llm_synthesis": bool(configuration.enable_insights_llm_synthesis),
	}

	payload_size = len(
		json.dumps(
			{"summary": summary, "items": findings, "metadata": metadata},
			ensure_ascii=True,
			default=str,
		)
	)
	metadata["full_payload_size_bytes"] = payload_size

	print(
		f"[InsightsGenerator] {len(findings)} findings generated from "
		f"{len(state.structured_results)} entities and {len(source_map)} sources"
	)

	return {
		"insights_summary": summary,
		"insights_items": findings,
		"insights_metadata": metadata,
	}

