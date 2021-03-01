import time
import numpy as np
import kinatt
from kinatt import ObjectiveFunction
from kinatt import PoseConstraint
from kinatt import Attractor
from kinatt import CollisionConstraint

import skrobot
from skrobot.models import PR2
from skrobot.model import Box
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config

table = Box(extents=[0.1, 0.1, 3.0], with_sdf=True)
table.translate([0.7, -0.45, 1.2])

mech = kinatt.create_pr2_mechanism()
with_base = True
pose_const = PoseConstraint(mech, "r_gripper_tool_frame", [0.2, -0.8, 0.6], with_base=with_base) 
objfun = ObjectiveFunction.from_constraint(pose_const)

colavoid = CollisionConstraint(mech, table.sdf, with_base=with_base)
attractor = Attractor(objfun, colavoid)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
robot = PR2()
joint_list = [robot.__dict__[name] for name in mech.joint_names]
viewer.add(robot)
viewer.add(table)
viewer.show()
joint_angles = np.zeros(17)
for i in range(200):
    joint_angles = attractor.propagate(joint_angles, radius=0.02)
    set_robot_config(robot, joint_list, joint_angles, with_base=with_base)
    viewer.redraw()
    time.sleep(0.1)

