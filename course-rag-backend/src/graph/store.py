import json
import networkx as nx
from pathlib import Path
from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

_G: nx.DiGraph = nx.DiGraph()


def load() -> None:
    """Load graph from graph_store.json on startup if file exists."""
    path = Path(settings.GRAPH_STORE_PATH)
    if not path.exists():
        log.info("No graph_store.json found — starting with empty graph.")
        return
    try:
        with open(path) as f:
            data = json.load(f)
        global _G
        try:
            _G = nx.node_link_graph(data, directed=True, multigraph=False, edges="links")
        except KeyError:
            _G = nx.node_link_graph(data, directed=True, multigraph=False, edges="edges")
        log.info(f"Graph loaded: {_G.number_of_nodes()} nodes, {_G.number_of_edges()} edges")
    except Exception as e:
        log.error(f"Failed to load graph: {e}")


def save() -> None:
    """Serialise graph to graph_store.json using node_link_data format."""
    path = Path(settings.GRAPH_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = nx.node_link_data(_G, edges="links")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Graph saved: {_G.number_of_nodes()} nodes, {_G.number_of_edges()} edges")
    except Exception as e:
        log.error(f"Failed to save graph: {e}")


def get_graph() -> nx.DiGraph:
    return _G


def reset_graph() -> None:
    """Clear the in-memory graph — used before re-ingestion."""
    global _G
    _G = nx.DiGraph()
    log.info("Graph reset.")