from __future__ import annotations
import mujoco
import numpy as np
import quaternion as qnp

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.lynx_arm.modules.module import Module
from ariel.body_phenotypes.lynx_arm.config import (
    EE_FLANGE_RADIUS, EE_FLANGE_THICK, EE_MASS, TCP_OFFSET
)

def _quat_x_deg(deg: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(deg), 0.0, 0.0])
    return np.roll(qnp.as_float_array(quat), shift=-1)

def _quat_xyz_deg(x: float, y: float, z: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(x), np.deg2rad(y), np.deg2rad(z)])
    return np.roll(qnp.as_float_array(quat), shift=-1)

class LynxEndEffector(Module):
    """Neutral flange with a tool center point (TCP) site."""

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


        root.add_geom(
            name=f"ee_flange_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[EE_FLANGE_RADIUS, EE_FLANGE_THICK/2, 0.0],
            quat=_quat_x_deg(90.0),   # along +Y
            mass=EE_MASS,
            rgba=(0.20, 0.20, 0.22, 1.0),
        )

        self.sites = {}
        self.sites[ModuleFaces.FRONT] = root.add_site(
            name=f"ee_mount_{index}",
            pos=[0.0, EE_FLANGE_THICK/2, 0.0],
        )

        # Tool center point forward
        self.tcp = root.add_site(
            name=f"ee_tcp_{index}",
            pos=[0.0, EE_FLANGE_THICK/2 + TCP_OFFSET, 0.0],
        )

        self.spec = spec
        self.body = root
        self.rotate(0.0)

    def rotate(self, angle: float) -> None:
        self.body.quat = np.round(_quat_xyz_deg(180.0, -(180.0 - angle), 0.0), 3)