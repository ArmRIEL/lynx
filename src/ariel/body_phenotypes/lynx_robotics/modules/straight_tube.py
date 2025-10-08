import numpy as np
from src.ariel.body_phenotypes.lynx_robotics.tools.module import Module, AttachmentPoint
from src.ariel.body_phenotypes.lynx_robotics.tools.unbuilt_child import UnbuiltChild
from src.ariel.body_phenotypes.lynx_robotics.tools.math_utils import Vector3, Quaternion, ensure_list, angle_to_quaternion

class StraightTube(Module):

    ATTACH = 0

    def __init__(
        self,
        cylinder_length : float = 0.5,
        cylinder_radius : float = 0.05,
        name: str = "tube",
    ):
        
        self.cylinder_length = cylinder_length
        self.cylinder_radius = cylinder_radius
        self.length = cylinder_length
        self.radius = cylinder_radius
        self.name = name if name is not None else "StraightTube"

        attachment_points = { 
            self.ATTACH: AttachmentPoint(
                offset=Vector3([0.0, 0.0, self.length/2]),
                orientation=Quaternion(),  # Rotate around the z-axis
            ),
        }

        super().__init__(attachment_points)

    @property
    def attach(self) -> Module:
        return self._children.get(self.ATTACH)

    @attach.setter
    def attach(self, module: Module) -> None:
        self.set_child(module, self.ATTACH)

    def build(self, mjcf, entry_point, attachment_point_pos, attachment_point_quat) -> list:

        # Convert attachment_point_pos to list if it's a Vector3 object
        pos_list = ensure_list(attachment_point_pos)

        # Build the module
        # Calculate the axis given the attachment point quaternion
        mj_quat1 = attachment_point_quat.to_mujoco_format()
        new_pos1 = Vector3([0,0,self.length/2])

        body = entry_point.add("body", name=self.name, pos=pos_list, quat=mj_quat1)
        body.add("geom", type="cylinder", size=[self.cylinder_radius, self.cylinder_length / 2], pos=ensure_list(new_pos1), rgba=[1,1,1,1], name="lynx_"+self.name) # 

        # Find the children
        tasks = []

        for child_index, attachment_point in self.attachment_points.items():
            child = self.children.get(child_index)
            if child is not None:
                unbuilt = UnbuiltChild(module=child, mj_body=body)
                unbuilt.make_pose(
                    position=Vector3([0, 0, self.cylinder_length]), # Prev pos + offset rotated in right direction = attachment point pos
                    orientation=Quaternion(),
                )
                tasks.append(unbuilt)

        return tasks