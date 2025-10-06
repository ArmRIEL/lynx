from __future__ import annotations
import mujoco
import numpy as np
import quaternion as qnp

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.lynx_arm.modules.module import Module

# Simple base plate dimensions (meters); adjust if you have exact CAD.
BASE_L = 0.160  # X
BASE_W = 0.225  # Z
BASE_T = 0.010  # Y
BASE_MASS = 0.88

def _quat_xyz_deg(x: float, y: float, z: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(x), np.deg2rad(y), np.deg2rad(z)])
    return np.roll(qnp.as_float_array(quat), shift=-1)

class LynxBase(Module):
    """Flat base plate (fixed)."""

    index: int | None = None
    module_type: str = ModuleType.BRICK

    def __init__(self, index: int, *,
                 spec: mujoco.MjSpec | None = None,
                 parent = None,
                 origin_pos=(0.0, 0.0, 0.0),
                 origin_quat=None) -> None:
        self.index = index
        spec = spec or mujoco.MjSpec()
        
        container = (parent or spec.worldbody)
        kwargs = dict(name=F"..._{index}", pos=list(origin_pos))
        if origin_quat is not None:
            kwargs["quat"] = list(origin_quat)
        root = container.add_body(**kwargs)

        # Box geom (MuJoCo uses half-extents for box size)
        root.add_geom(
            name=f"base_box_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[BASE_L/2, BASE_T/2, BASE_W/2],
            mass=BASE_MASS,
            rgba=(0.12, 0.12, 0.13, 1.0),
        )

        # Front site at the top center (+Y)
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = root.add_site(
            name=f"base_front_{index}",
            pos=[0.0, BASE_T/2, 0.0],
        )

        self.spec = spec
        self.body = root
        self.rotate(0.0)

    def rotate(self, angle: float) -> None:
        self.body.quat = np.round(_quat_xyz_deg(180.0, -(180.0 - angle), 0.0), 3)