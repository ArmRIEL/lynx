from __future__ import annotations
import mujoco
from typing import Dict, List

def list_joint_names(model: mujoco.MjModel) -> List[str]:
    return [model.id2name(i, mujoco.mjtObj.mjOBJ_JOINT) for i in range(model.njnt)]

def list_actuator_names(model: mujoco.MjModel) -> List[str]:
    return [model.id2name(i, mujoco.mjtObj.mjOBJ_ACTUATOR) for i in range(model.nu)]

def site_name_to_id(model: mujoco.MjModel) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sid in range(model.nsite):
        name = model.id2name(sid, mujoco.mjtObj.mjOBJ_SITE)
        if name:
            out[name] = sid
    return out

def find_tcp_site_id(model: mujoco.MjModel) -> int | None:
    for sid in range(model.nsite):
        name = model.id2name(sid, mujoco.mjtObj.mjOBJ_SITE)
        if name and "ee_tcp_" in name:
            return sid
    return None