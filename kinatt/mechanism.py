import uuid
import numpy as np
from skrobot.models import PR2
from skrobot.planner.swept_sphere import compute_swept_sphere
import tinyfk

class Mechanism(object):

    def __init__(self, robot_model, fksolver, joint_names, col_link_names):
        """
        skrobot's robot_model will only be used in this constructor.
        In future release, probably, this dependency on skrobot will be removed.
        """
        self.dof = len(joint_names)
        self.fksolver = fksolver
        self.joint_names = joint_names
        self.joint_ids = self.fksolver.get_joint_ids(joint_names)
        self.col_link_ids = self.fksolver.get_link_ids(col_link_names)

        sphere_ids, radius_list = self._compute_collision_features(robot_model, col_link_names)
        self.sphere_ids = sphere_ids
        self.radius_list = radius_list

    def _compute_collision_features(self, robot_model, col_link_names):
        """ 
        called only in constructor
        """
        sphere_name_list = []
        radius_list = []
        for col_link_name in col_link_names:
            coll_link_id = self.fksolver.get_link_ids([col_link_name])[0]
            col_mesh = robot_model.__dict__[col_link_name].collision_mesh
            centers, R = compute_swept_sphere(col_mesh)

            radius_list.extend([R for _ in range(len(centers))])
            for center in centers:
                sphere_name = str(uuid.uuid1())
                self.fksolver.add_new_link(sphere_name, coll_link_id, center)
                sphere_name_list.append(sphere_name)

        sphere_ids = self.fksolver.get_link_ids(sphere_name_list)
        return sphere_ids, np.array(radius_list)

    def forward_kinematics(self, joint_angles, link_names,
            with_rot=False, with_base=False, with_jacobian=False):
        link_names = listify(link_names)
        link_ids = self.fksolver.get_link_ids(link_names)
        return self._forward_kinematics(joint_angles, link_ids, 
                with_base=with_base, with_jacobian=with_jacobian)

    def _forward_kinematics(self, joint_angles, link_ids,
            with_rot=False, with_base=False, with_jacobian=False):
        P, J = self.fksolver.solve_forward_kinematics(
                [joint_angles], link_ids, self.joint_ids, 
                with_rot=with_rot, with_base=with_base, with_jacobian=with_jacobian)
        return P, J

    def collision_forward_kinematics(self, joint_angles, with_base=False, with_jacobian=False):
        return self._forward_kinematics(joint_angles, self.sphere_ids,
                with_base=with_base, with_jacobian=with_jacobian)

def create_pr2_mechanism():
    robot_model = PR2()
    fksolver = tinyfk.RobotModel(robot_model.urdf_path)

    joint_names = [
            "r_shoulder_pan_joint", "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint", "r_elbow_flex_joint",
            "r_forearm_roll_joint", "r_wrist_flex_joint",
            "r_wrist_roll_joint",
            "l_shoulder_pan_joint", "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint", "l_elbow_flex_joint",
            "l_forearm_roll_joint", "l_wrist_flex_joint",
            "l_wrist_roll_joint"
            ]
    col_link_names = [
            "head_tilt_link",
            "r_shoulder_lift_link", "l_shoulder_lift_link",
            "base_link",
            "r_upper_arm_link", "r_forearm_link",
            "r_gripper_palm_link", 
            "l_upper_arm_link", "l_forearm_link",
            "l_gripper_palm_link"
            ]
    return Mechanism(robot_model, fksolver, joint_names, col_link_names)
