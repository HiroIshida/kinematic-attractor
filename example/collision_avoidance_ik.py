import time
import numpy as np
import kinatt
from kinatt import ObjectiveFunction
from kinatt import PoseConstraint
from kinatt import Attractor

import skrobot
from skrobot.models import PR2
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config

mech = kinatt.create_pr2_mechanism()
with_base = False
pose_const = PoseConstraint(mech, "r_gripper_tool_frame", [0.2, -0.6, 0.6], with_base=with_base) 
objfun = ObjectiveFunction.from_constraint(pose_const)
attractor = Attractor(objfun)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
robot = PR2()
joint_list = [robot.__dict__[name] for name in mech.joint_names]
viewer.add(robot)
viewer.show()
joint_angles = np.zeros(14)
for i in range(100):
    joint_angles = attractor.propagate(joint_angles)
    set_robot_config(robot, joint_list, joint_angles, with_base=with_base)
    viewer.redraw()
    time.sleep(0.1)


