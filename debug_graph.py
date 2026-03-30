
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import crawler.graph...")
    from crawler.graph import graph
    print("Successfully imported 'graph' from 'crawler.graph'")
    print(f"Graph name: {getattr(graph, 'name', 'N/A')}")
    print("Graph is a CompiledGraph:", hasattr(graph, 'get_graph'))
except Exception as e:
    print(f"Error importing graph: {e}")
    import traceback
    traceback.print_exc()
