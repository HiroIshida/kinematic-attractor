import kinatt
from kinatt import ObjectiveFunction
from kinatt import PoseConstraint

mech = kinatt.create_pr2_mechanism()
pose_const = PoseConstraint(mech, "r_gripper_tool_frame", [0.7, -0.5, 0.7], with_base=False) 
objfun = ObjectiveFunction.from_constraint(pose_const)
