from __future__ import annotations
import mujoco
import numpy as np
import quaternion as qnp

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.lynx_arm.modules.module import Module
from ariel.body_phenotypes.lynx_arm.config import (
    CF_TUBE_RAD, CLAMP_RING_LEN, STATOR_MASS, ROTOR_MASS, SHRINK,
    DEFAULT_KP, DEFAULT_KV, DEFAULT_CTRL_RANGE
)

# --- helpers -----------------------------------------------------------------
def _quat_from_xyz_deg(x: float, y: float, z: float) -> np.ndarray:
    quat = qnp.from_euler_angles([np.deg2rad(x), np.deg2rad(y), np.deg2rad(z)])
    return np.roll(qnp.as_float_array(quat), shift=-1)

def _rot_x_deg(deg: float) -> np.ndarray:
    return _quat_from_xyz_deg(deg, 0.0, 0.0)

def _set_body_quat(body: mujoco.MjBody, x: float, y: float, z: float) -> None:
    body.quat = np.round(_quat_from_xyz_deg(x, y, z), 3)

def _add_affine_servo(
    spec: mujoco.MjSpec,
    rotor: mujoco.MjBody,
    name: str,
    axis=(0, 0, 1),
    pos=(0, 0, 0),
    kp: float = DEFAULT_KP,
    kv: float = DEFAULT_KV,
    ctrl_range=DEFAULT_CTRL_RANGE,
    armature: float | None = None,
    damping: float | None = None,
    frictionloss: float | None = None,
) -> None:
    rotor.add_joint(
        name=name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=axis,
        pos=pos,
        armature=armature if armature is not None else 0.0,
        damping=damping if damping is not None else 0.0,
        frictionloss=frictionloss if frictionloss is not None else 0.0,
    )

    dynprm = np.zeros(10)
    gainprm = np.zeros(10)
    biasprm = np.zeros(10)
    gainprm[0] = kp
    biasprm[:3] = [0.0, -kp, -kv]

    spec.add_actuator(
        name=name,
        dyntype=mujoco.mjtDyn.mjDYN_NONE,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        dynprm=dynprm, gainprm=gainprm, biasprm=biasprm,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=name,
        ctrlrange=ctrl_range,
    )


class LynxHingeOrthogonal(Module):
    """SES-PRO hinge with tube mounted 90° orthogonal to inline."""

    index: int | None = None
    module_type: str = ModuleType.HINGE

    def __init__(self, index: int, *,
                 tube_length: float = 0.118,
                 clamp_length: float = CLAMP_RING_LEN,
                 kp: float = DEFAULT_KP, kv: float = DEFAULT_KV,
                 ctrl_range=DEFAULT_CTRL_RANGE,
                 armature: float | None = None,
                 damping: float | None = None,
                 frictionloss: float | None = None,
                 spec: mujoco.MjSpec | None = None,
                 parent: mujoco.MjBody | None = None,
                 origin_pos=(0.0, 0.0, 0.0),
                 origin_quat=None) -> None:
        self.index = index

        stator_hlen = (clamp_length * SHRINK) * 0.5
        rotor_hlen  = (tube_length  * SHRINK) * 0.5

        spec = spec or mujoco.MjSpec()
        
        container = (parent or spec.worldbody)
        kwargs = dict(name=f"lynx_hinge_inline_{index}", pos=list(origin_pos))
        if origin_quat is not None:
            kwargs["quat"] = list(origin_quat)
        hinge = container.add_body(**kwargs)

        # Stator along +Y
        stator = hinge.add_body(name=f"stator_{index}", pos=[0.0, stator_hlen, 0.0])
        stator.add_geom(
            name=f"stator_cyl_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[CF_TUBE_RAD, stator_hlen, 0.0],
            quat=_rot_x_deg(90.0),
            mass=STATOR_MASS,
            rgba=(0.18, 0.18, 0.20, 1.0),
        )

        # Mount rotated +90° about X so rotor local +Y => world +Z
        rotor_mount = hinge.add_body(
            name=f"rotor_mount_{index}",
            pos=[0.0, (clamp_length * SHRINK) + rotor_hlen, 0.0],
        )
        _set_body_quat(rotor_mount, 90.0, 0.0, 0.0)

        # Rotor tube (local +Y)
        rotor = rotor_mount.add_body(name=f"rotor_{index}", pos=[0.0, 0.0, 0.0])
        rotor.add_geom(
            name=f"rotor_tube_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[CF_TUBE_RAD, rotor_hlen, 0.0],
            quat=_rot_x_deg(90.0),
            mass=ROTOR_MASS,
            rgba=(0.05, 0.05, 0.06, 1.0),
        )

        # Attachment site at tube tip (rotor local +Y)
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = rotor.add_site(
            name=f"lynx_orth_front_{index}",
            pos=[0.0, rotor_hlen, 0.0],
        )

        # Joint + actuator
        _add_affine_servo(
            spec=spec,
            rotor=rotor,
            name=f"servo_orth_{index}",
            axis=(0, 0, 1),
            pos=[0.0, -rotor_hlen, 0.0],
            kp=kp, kv=kv, ctrl_range=ctrl_range,
            armature=armature, damping=damping, frictionloss=frictionloss,
        )

        spec.add_exclude(bodyname1=stator.name, bodyname2=rotor.name)

        self.spec = spec
        self.body = hinge
        self.rotate(0.0)

    def rotate(self, angle: float) -> None:
        self.body.quat = np.round(_quat_from_xyz_deg(180.0, -(180.0 - angle), 0.0), 3)
