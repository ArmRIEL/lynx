import mujoco
import mujoco.viewer
import numpy as np

from src.ariel.body_phenotypes.lynx_robotics.lynx_manipulator import lynx_manipulator
from src.ariel.body_phenotypes.lynx_robotics.tools.build_file import build_mjcf
from src.ariel.body_phenotypes.lynx_robotics.tools.mj_default_sim_setup import mujoco_setup_sim
from src.ariel.body_phenotypes.lynx_robotics.scenes.table import table_terrain


robot_description = {
    "l_link2": 0.2805,
    "l_link3": 0.3055,
    "joint_types": ["inline", "inline", "orthogonal", "orthogonal", "orthogonal", "orthogonal"],
    "joint_angles": [0, -np.pi, 0, np.pi, 0, 0],
    "control_mode": "position",
}


def main():
    # Assemble the Lynx manipulator using your modular definition
    body = lynx_manipulator()

    # Convert the hierarchical model into MuJoCo XML
    xml_string = build_mjcf(
        bodies=[body],                          # one robot (the Lynx)
        body_poss=[[0, 0, 0.825]],              # position of base in world
        body_oris=[[0, 0, 0, 1]],               # orientation (quaternion)
        terrain_builder=table_terrain,          # add a simple flat table
        sim_setup=mujoco_setup_sim,             # default simulation setup
        ts=0.002,                               # timestep = 2 ms
    )

    # 3Load model and data into MuJoCo
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Launch MuJoCo passive viewer to visualize the robot
    viewer = mujoco.viewer.launch_passive(model, data)

    print("MuJoCo viewer launched. Close the window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)  # advance simulation (no control)
        viewer.sync()

    print("Viewer closed — exiting.")


if __name__ == "__main__":
    main()
