"""Decoders that turn configs into a Lynx phenotype graph (NetworkX DiGraph)."""

from __future__ import annotations
import networkx as nx
from networkx.readwrite import json_graph
from typing import List, Dict, Any

def decode_lynx_config(
    joint_kinds: List[str],      # e.g. ["inline","orthogonal"]
    tube_lengths: List[float],   # meters, one per tube in order
    twists_deg: List[float] | None = None,
    add_end_effector: bool = True,
) -> nx.DiGraph:
    """
    Returns a rooted, oriented tree:
      0(base) -> 1(joint) -> 2(tube) -> 3(joint) -> 4(tube) -> [5(ee)]
    Node attrs:
      type: "base"|"hinge_inline"|"hinge_orthogonal"|"tube"|"end_effector"
      length: (for tubes)
      twist_deg: fixed assembly twist applied at parent->child interface
    """
    G = nx.DiGraph()
    twists = twists_deg or []

    # 0: base
    G.add_node(0, type="base")
    # 1: first joint
    j0 = joint_kinds[0].lower()
    G.add_node(1, type=("hinge_inline" if j0 in ("inline","in") else "hinge_orthogonal"),
               twist_deg=(twists[0] if len(twists) > 0 else 0.0))
    # 2: tube1
    G.add_node(2, type="tube", length=float(tube_lengths[0]),
               twist_deg=(twists[1] if len(twists) > 1 else 0.0))
    # 3: second joint
    j1 = joint_kinds[1].lower()
    G.add_node(3, type=("hinge_inline" if j1 in ("inline","in") else "hinge_orthogonal"),
               twist_deg=(twists[2] if len(twists) > 2 else 0.0))
    # 4: tube2
    G.add_node(4, type="tube", length=float(tube_lengths[1]),
               twist_deg=(twists[3] if len(twists) > 3 else 0.0))
    # 5: end effector (optional)
    if add_end_effector:
        G.add_node(5, type="end_effector",
                   twist_deg=(twists[4] if len(twists) > 4 else 0.0))

    # Edges (always FRONT in this phenotype)
    G.add_edge(0, 1, face="front")
    G.add_edge(1, 2, face="front")
    G.add_edge(2, 3, face="front")
    G.add_edge(3, 4, face="front")
    if add_end_effector:
        G.add_edge(4, 5, face="front")

    return G


# ---------------------------- JSON helpers -----------------------------------
def to_tree_json(G: nx.DiGraph, root: int = 0) -> Dict[str, Any]:
    """Serialize to JSON-able tree data (NetworkX json_graph.tree_data)."""
    return json_graph.tree_data(G, root=root)

def from_tree_json(data: Dict[str, Any]) -> nx.DiGraph:
    """Deserialize tree JSON back to a DiGraph (json_graph.tree_graph)."""
    return json_graph.tree_graph(data)
