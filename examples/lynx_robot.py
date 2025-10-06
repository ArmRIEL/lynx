# examples/view_lynx.py
import mujoco
import mujoco.viewer as mjv

from ariel.body_phenotypes.lynx_arm.prebuilt_robots.lynx_arm_5part import build

if __name__ == "__main__":
    model, data, parts = build()

    # Quick, blocking viewer (MuJoCo manages the timing/loop):
    # mjv.launch(model, data)  # uncomment this line to use the blocking viewer

    # Or: passive viewer where you control the stepping yourself:
    with mjv.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
