from __future__ import annotations

import math
import numpy as np
import mujoco
from typing import Dict, Tuple

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces  # enum reuse

from ariel.body_phenotypes.lynx_arm.modules import (
    LynxBase, LynxHingeInline, LynxHingeOrthogonal, LynxTube, LynxEndEffector
)
from ariel.body_phenotypes.lynx_arm.config import CF_LEN_280_5, CF_LEN_143


# ---------------------------- quaternion helpers -----------------------------
def _axis_angle_xyzw(axis: Tuple[float, float, float], angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n == 0:
        return np.array([0, 0, 0, 1], float)
    ax /= n
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return np.array([ax[0]*s, ax[1]*s, ax[2]*s, c], float)

def _compose_quat_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], float)

def _site_local_pose(site: mujoco.MjSite) -> Tuple[np.ndarray, np.ndarray]:
    # MjSpec sites store local position; we use identity orientation for sites.
    return np.array(site.pos, float), np.array([0, 0, 0, 1], float)


# ---------------------------- low-level attach -------------------------------
def _attach(parent_body: mujoco.MjBody,
            parent_site: mujoco.MjSite,
            ctor, *, spec: mujoco.MjSpec, index: int,
            twist_deg: float = 0.0, **kwargs):
    """
    Attach a child Module at the parent's FRONT site with an optional fixed
    'assembly' twist around the site forward axis (+Y in our modules).
    """
    pos, site_q = _site_local_pose(parent_site)
    twist_q = _axis_angle_xyzw((0, 1, 0), math.radians(twist_deg))
    origin_q = _compose_quat_xyzw(twist_q, site_q)
    return ctor(index=index, spec=spec, parent=parent_body,
                origin_pos=pos, origin_quat=origin_q, **kwargs)


# ------------------------------ builders -------------------------------------
def build_lynx_5part(tube1_len: float = CF_LEN_280_5,
                     tube2_len: float = CF_LEN_143):
    """Base -> JointInline -> Tube -> JointOrthogonal -> Tube -> EndEffector"""
    spec = mujoco.MjSpec()

    base = LynxBase(index=0, spec=spec)
    j1   = _attach(base.body, base.sites[ModuleFaces.FRONT], LynxHingeInline,    spec=spec, index=1)
    t1   = _attach(j1.body,   j1.sites[ModuleFaces.FRONT],   LynxTube,           spec=spec, index=2, length=tube1_len)
    j2   = _attach(t1.body,   t1.sites[ModuleFaces.FRONT],   LynxHingeOrthogonal,spec=spec, index=3)
    t2   = _attach(j2.body,   j2.sites[ModuleFaces.FRONT],   LynxTube,           spec=spec, index=4, length=tube2_len)
    ee   = _attach(t2.body,   t2.sites[ModuleFaces.FRONT],   LynxEndEffector,    spec=spec, index=5)

    for m in (base, j1, t1, j2, t2, ee):
        m.rotate(0.0)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, dict(base=base, j1=j1, t1=t1, j2=j2, t2=t2, ee=ee)


def build_lynx_configurable(
    *,
    joint_kinds,     # list[str]: e.g., ["inline", "orthogonal"]
    tube_lengths,    # list[float]: meters for each tube in order
    twists_deg=None, # list[float]: optional fixed assembly twists per connection
    add_end_effector: bool = True,
):
    """Configurable assembly using the same attach logic."""
    spec = mujoco.MjSpec()
    twists = twists_deg or []

    parts: Dict[str, object] = {}
    conn_i = 0

    # Base
    base = LynxBase(index=0, spec=spec)
    parts["base"] = base

    # J1
    J1Ctor = LynxHingeInline if joint_kinds[0].lower() in ("inline", "in") else LynxHingeOrthogonal
    j1 = _attach(base.body, base.sites[ModuleFaces.FRONT], J1Ctor, spec=spec,
                 index=1, twist_deg=(twists[conn_i] if conn_i < len(twists) else 0.0))
    parts["j1"] = j1; conn_i += 1

    # Tube 1
    t1 = _attach(j1.body, j1.sites[ModuleFaces.FRONT], LynxTube, spec=spec, index=2,
                 length=tube_lengths[0], twist_deg=(twists[conn_i] if conn_i < len(twists) else 0.0))
    parts["t1"] = t1; conn_i += 1

    # J2
    J2Ctor = LynxHingeInline if joint_kinds[1].lower() in ("inline", "in") else LynxHingeOrthogonal
    j2 = _attach(t1.body, t1.sites[ModuleFaces.FRONT], J2Ctor, spec=spec,
                 index=3, twist_deg=(twists[conn_i] if conn_i < len(twists) else 0.0))
    parts["j2"] = j2; conn_i += 1

    # Tube 2
    t2 = _attach(j2.body, j2.sites[ModuleFaces.FRONT], LynxTube, spec=spec, index=4,
                 length=tube_lengths[1], twist_deg=(twists[conn_i] if conn_i < len(twists) else 0.0))
    parts["t2"] = t2; conn_i += 1

    if add_end_effector:
        ee = _attach(t2.body, t2.sites[ModuleFaces.FRONT], LynxEndEffector, spec=spec, index=5,
                     twist_deg=(twists[conn_i] if conn_i < len(twists) else 0.0))
        parts["ee"] = ee

    for m in parts.values():
        m.rotate(0.0)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, parts


# ------------------------------ graph builder --------------------------------
def build_from_graph(G) -> Tuple[mujoco.MjModel, mujoco.MjData, Dict[int, object]]:
    """
    Build from a decoded NetworkX DiGraph.
    Node attrs:
      - type: "base"|"hinge_inline"|"hinge_orthogonal"|"tube"|"end_effector"
      - length: for tubes (m)
      - twist_deg: fixed assembly twist at the parent->child interface
    Edges:
      - face: default "front"
    """
    import networkx as nx
    assert isinstance(G, nx.DiGraph), "Expected a NetworkX DiGraph."

    spec = mujoco.MjSpec()
    parts: Dict[int, object] = {}

    # Find root (in-degree 0)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    assert len(roots) == 1, "Graph must be a single rooted tree."
    root = roots[0]

    def ctor_for(node_type: str):
        match node_type:
            case "base":           return LynxBase
            case "hinge_inline":   return LynxHingeInline
            case "hinge_orthogonal": return LynxHingeOrthogonal
            case "tube":           return LynxTube
            case "end_effector":   return LynxEndEffector
            case _:
                raise ValueError(f"Unknown node type: {node_type}")

    # Build root under world
    ntype = G.nodes[root].get("type")
    parts[root] = ctor_for(ntype)(index=int(root), spec=spec)

    # BFS over tree
    for parent, child, edata in nx.bfs_edges(G, root):
        face = edata.get("face", "front")

        # parent attachment site (we only use FRONT in this phenotype)
        parent_site = parts[parent].sites[ModuleFaces.FRONT]

        # child ctor + kwargs
        ch_attrs = G.nodes[child]
        twist = float(ch_attrs.get("twist_deg", 0.0))
        ctor = ctor_for(ch_attrs["type"])
        kwargs = {"index": int(child)}

        if ch_attrs["type"] == "tube":
            kwargs["length"] = float(ch_attrs["length"])

        parts[child] = _attach(parts[parent].body, parent_site, ctor, spec=spec,
                               twist_deg=twist, **kwargs)

    for m in parts.values():
        m.rotate(0.0)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, parts
