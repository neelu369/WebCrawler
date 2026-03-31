"""
Adapter to integrate incubator-specific flow into general LangGraph pipeline.

This allows the existing pipeline to handle incubator discovery
while adding specialized nodes for the specific use case.
"""

from __future__ import annotations

from typing import Literal
from langgraph.graph import StateGraph

from crawler.config import Configuration
from crawler.state import InputState, OutputState, State

# Import incubator-specific nodes
from crawler.nodes.incubator_discovery_node import discover_incubators, enrich_incubator_entity
from crawler.nodes.web_crawler import crawl_pages
from crawler.nodes.source_verifier import verify_sources
from crawler.nodes.mongo_logger import log_to_mongo
from crawler.nodes.preprocessor import preprocess
from crawler.nodes.graph_structurer import structure_from_graph
from crawler.nodes.insights_generator import generate_insights
from crawler.nodes.metrics_evaluator import evaluate_metrics
from crawler.routing import route_after_evaluation


def build_incubator_pipeline():
    """
    Build a specialized LangGraph for Indian incubator discovery.
    
    Flow:
      START → incubator_discovery → web_crawler → source_verifier
            → mongo_logger → enrich_incubator → graph_structurer
            → insights_generator → metrics_evaluator → END
    """
    
    workflow = StateGraph(
        State,
        input=InputState,
        output=OutputState,
        config_schema=Configuration,
    )
    
    # Add nodes
    workflow.add_node("incubator_discovery", discover_incubators)
    workflow.add_node("web_crawler", crawl_pages)
    workflow.add_node("source_verifier", verify_sources)
    workflow.add_node("mongo_logger", log_to_mongo)
    workflow.add_node("preprocessor", preprocess)
    workflow.add_node("enrich_incubator", enrich_incubator_entity)
    workflow.add_node("graph_structurer", structure_from_graph)
    workflow.add_node("insights_generator", generate_insights)
    workflow.add_node("metrics_evaluator", evaluate_metrics)
    
    # Define edges
    workflow.add_edge("__start__", "incubator_discovery")
    workflow.add_edge("incubator_discovery", "web_crawler")
    workflow.add_edge("web_crawler", "source_verifier")
    workflow.add_edge("source_verifier", "mongo_logger")
    workflow.add_edge("mongo_logger", "preprocessor")
    workflow.add_edge("preprocessor", "enrich_incubator")
    workflow.add_edge("enrich_incubator", "graph_structurer")
    workflow.add_edge("graph_structurer", "insights_generator")
    workflow.add_edge("insights_generator", "metrics_evaluator")
    workflow.add_conditional_edges("metrics_evaluator", route_after_evaluation)
    
    return workflow.compile()


# Instance for use in main.py
incubator_graph = build_incubator_pipeline()
incubator_graph.name = "IndianIncubatorDiscoveryPipeline"
