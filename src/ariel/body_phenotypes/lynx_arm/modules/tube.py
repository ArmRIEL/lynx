from __future__ import annotations
import mujoco
import numpy as np
import quaternion as qnp

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.lynx_arm.modules.module import Module
from ariel.body_phenotypes.lynx_arm.config import CF_TUBE_RAD, TUBE_MASS

def _quat_x_deg(deg: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(deg), 0.0, 0.0])
    return np.roll(qnp.as_float_array(quat), shift=-1)

def _quat_xyz_deg(x: float, y: float, z: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(x), np.deg2rad(y), np.deg2rad(z)])
    return np.roll(qnp.as_float_array(quat), shift=-1)

class LynxTube(Module):
    """Straight SES-PRO tube (Ø70 mm OD) as a rigid link."""

    index: int | None = None
    module_type: str = ModuleType.BRICK

    def __init__(self, index: int, *,
                 length: float,
                 spec: mujoco.MjSpec | None = None,
                 parent = None,
                 origin_pos=(0.0, 0.0, 0.0),
                 origin_quat=None) -> None:
        self.index = index

        hlen = length * 0.5
        spec = spec or mujoco.MjSpec()

        container = (parent or spec.worldbody)
        kwargs = dict(name=F"..._{index}", pos=list(origin_pos))
        if origin_quat is not None:
            kwargs["quat"] = list(origin_quat)
        root = container.add_body(**kwargs)


        root.add_geom(
            name=f"tube_cyl_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[CF_TUBE_RAD, hlen, 0.0],      # [radius, half_length]
            quat=_quat_x_deg(90.0),        # orient cyl along +Y
            mass=TUBE_MASS,
            rgba=(0.06, 0.06, 0.07, 1.0),
        )

        self.sites = {}
        self.sites[ModuleFaces.FRONT] = root.add_site(
            name=f"tube_front_{index}",
            pos=[0.0, hlen, 0.0],          # tube tip (+Y)
        )

        self.spec = spec
        self.body = root
        self.rotate(0.0)

    def rotate(self, angle: float) -> None:
        self.body.quat = np.round(_quat_xyz_deg(180.0, -(180.0 - angle), 0.0), 3)
