import numpy as np
from utils import listify

class Constraint(object):
    def __init__(self, fun, n_state, m_constraint, is_equality, with_check=True):
        self.is_equality = is_equality
        self.n_state = n_state
        self.m_constraint = m_constraint
        self.fun = fun
        if with_check:
            self.check()

    def check(self):
        f, jac = self.fun(np.zeros(self.n_state))
        assert f.shape == (self.m_constraint,) 
        assert jac.shape == (self.m_constraint, self.n_state)

    def export_scipy(self):
        closure_member = {'jac_cache': None}

        def fun_scipinized(x):
            f, jac = self.fun(x)
            closure_member['jac_cache'] = jac
            return f

        def fun_scipinized_jac(x):
            return closure_member['jac_cache']

        return fun_scipinized, fun_scipinized_jac

    @classmethod
    def from_constraint_list(cls, constraint_list):
        is_all_false_or_all_true = (len(set([e.is_equality for e in constraint_list])) == 1)
        assert is_all_false_or_all_true

        n_state_combined = sum([e.n_state for e in constraint_list])
        m_constraint_combined = sum([e.m_constraint for e in constraint_list])

        def fun_combined_constraint(joint_angles):
            fs = []
            jacs = []
            for constraint in constraint_list:
                f, jac = constraint(joint_angles)
                fs.append(f)
                jacs.append(jac)
            f_combined = np.hstack(fs)
            jacs_combined = np.vstack(jacs)
            return f_combined, jacs_combined

        return cls(fun_combined_constraint, n_state_combined, m_constraint_combined,
                constraint_list[0].is_equality)

class ObjectiveFunction(object):
    def __init__(self, fun, n_state, with_check=True):
        self.fun = fun
        self.n_state = n_state
        if with_check:
            self.check()

    def check(self):
        f, grad = self.fun(np.zeros(self.n_state))
        assert isinstance(f, float)
        assert grad.shape == (self.n_state,)

    @classmethod
    def from_constraint(cls, constraint, weights=None):
        if weights is None:
            weights = np.ones(constraint.m_constraint)

        def fun(joint_angles):
            f, jac = constraint.fun(joint_angles)
            cost = np.sum((f * weights)**2) # (f w )^T (f w)
            grad = (2 * (weights ** 2) * f).dot(jac)
            return cost, grad

        return cls(fun, constraint.n_state)

    """
    @classmethod
    def from_constraint_list(cls, constraint_list, weights_list=None):
        is_all_equality = np.all([e.is_equality for e in  constraint_list])
        assert is_all_equality

        if weights_list is None:
            weights_list = [np.ones(e.m_constraint) for e in constraint_list
        if weights is None:
            weights = ones(len(constraint_list))
        assert len(weights) == len(constraint_list)
    """


class PoseConstraint(Constraint):
    def __init__(self, mechanism, link_name, pose_desired, with_base=False):

        with_rot = (len(pose_desired) != 3)
        link_ids = mechanism.fksolver.get_link_ids([link_name])
        n_state = mechanism.dof + with_base * 3
        m_constraint = len(pose_desired)

        def fun(joint_angles):
            P, J = mechanism._forward_kinematics(joint_angles, link_ids,
                    with_rot=with_rot, with_base=with_base, with_jacobian=True)
            diff = P[0] - pose_desired
            return diff, J

        super(PoseConstraint, self).__init__(
                fun, n_state, m_constraint, True, with_check=True)

"""
class StaticCollisionConstraint(Constraint):
    def __init__(self, mechanism, signed_distance_function):

        def fun(joint_angles):
            P, J = mechanism._forward_kinematics(joint_angles, link_ids,
                    with_rot, with_base, with_jacobian=True)
"""













