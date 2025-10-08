from pyrr import Quaternion
import numpy as np
import copy
from .modules.base import Base
from .modules.joint import JointInline, JointOrphogonal
from .modules.straight_tube import StraightTube
from .modules.right_angle_tube import RightAngleTube
from .modules.end_effector import EndEffector
from .modules.bezier_tube import BezierTube

from src.ariel.body_phenotypes.lynx_robotics.tools.math_utils import Vector3

# Nominal approximate lengths for conceptual links, corresponding to StraightTube definitions below
# These values are used where a fixed length is required for calculations (e.g., safety watchdog)
NOMINAL_LINK2_LENGTH_M = 0.2805
NOMINAL_LINK3_LENGTH_M = 0.3055
CONTROL_MODE = "position"  # Default control mode for joints

def lynx_manipulator(
    robot_description_dict: dict = {
        "l_link2": 0.2805,
        "l_link3": 0.3055,
        "joint_types": ["inline", "inline", "orthogonal", "orthogonal", "orthogonal", "orthogonal"],
        "joint_angles": [0, -np.pi, 0, np.pi, 0, 0],  # defaults from original function
        "control_mode": "position",
    }
):
    """Build a Lynx manipulator with variable joint types, link lengths, and joint angles."""

    # --- Helper to choose joint type ---
    def make_joint(joint_type, name, **kwargs):
        joint_class = JointOrphogonal if joint_type == "orthogonal" else JointInline
        return joint_class(name=name, **kwargs)

    # --- Base ---
    root = Base(
        base_length1=0.021,
        base_radius1=0.08,
        base_length2=0.021 + 0.001,
        base_radius2=0.062,
        name="lynx_base",
    )

    # --- Extract parameters ---
    cmode = robot_description_dict.get("control_mode", CONTROL_MODE)
    jt = robot_description_dict.get("joint_types", ["inline"] * 6)
    ja = robot_description_dict.get("joint_angles", [0, -np.pi, 0, np.pi, 0, 0])
    l2 = robot_description_dict.get("l_link2", 0.2805)
    l3 = robot_description_dict.get("l_link3", 0.3055)

    # --- Joints and links (same dimensions as original) ---
    j1 = make_joint(
        jt[0],
        "joint1",
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=(0.013 + 0.035) * 2 + 0.001,
        cylinder_radius2=0.062,
        angle=ja[0],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    j2 = make_joint(
        jt[1],
        "joint2",
        cylinder_length1=0.13,
        cylinder_radius1=0.062,
        cylinder_length2=0.013 * 2,
        cylinder_radius2=0.062,
        angle=ja[1],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    st1 = StraightTube(
        cylinder_length=l2 + 0.008,
        cylinder_radius=0.042,
        name="link2_tube",
    )

    j3 = make_joint(
        jt[2],
        "joint3",
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.029,
        cylinder_radius2=0.042,
        angle=ja[2],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    j4 = make_joint(
        jt[3],
        "joint4",
        cylinder_length1=0.09218,
        cylinder_radius1=0.042,
        cylinder_length2=0.029,
        cylinder_radius2=0.042,
        angle=ja[3],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    st2 = StraightTube(
        cylinder_length=l3 + 0.008,
        cylinder_radius=0.042,
        name="link3_tube",
    )

    j5 = make_joint(
        jt[4],
        "joint5",
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.027,
        cylinder_radius2=0.042,
        angle=ja[4],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    j6 = make_joint(
        jt[5],
        "joint6",
        cylinder_length1=0.096,
        cylinder_radius1=0.042,
        cylinder_length2=0.027,
        cylinder_radius2=0.042,
        angle=ja[5],
        control_mode=cmode,
        armature=0.01, damping=100000, frictionloss=1e-6,
    )

    ee_cyl = StraightTube(
        cylinder_length=0.05,
        cylinder_radius=0.042,
        name="ee_cylinder",
    )

    ee = EndEffector(name="end_effector")

    # --- Chain assembly ---
    root.attach = j1
    j1.attach = j2
    j2.attach = st1
    st1.attach = j3
    j3.attach = j4
    j4.attach = st2
    st2.attach = j5
    j5.attach = j6
    j6.attach = ee_cyl
    ee_cyl.attach = ee

    return root
