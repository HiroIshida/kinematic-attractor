import copy
import numpy as np
import scipy
from utils import listify
from utils import scipinize

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
        scifun, scijac = scipinize(self.fun)
        cstr_type = 'eq' if self.is_equality else 'ineq'
        dic = {'type': cstr_type, 'fun': scifun,
                     'jac': scijac}
        return dic

    @classmethod
    def from_constraint_list(cls, constraint_list):
        is_all_false_or_all_true = (len(set([e.is_equality for e in constraint_list])) == 1)
        assert is_all_false_or_all_true

        is_same_state_number = (len(set([e.n_state for e in constraint_list])) == 1)
        assert is_same_state_number

        m_constraint_combined = sum([e.m_constraint for e in constraint_list])

        def fun_combined_constraint(joint_angles):
            fs = []
            jacs = []
            for constraint in constraint_list:
                f, jac = constraint.fun(joint_angles)
                fs.append(f)
                jacs.append(jac)
            f_combined = np.hstack(fs)
            jacs_combined = np.vstack(jacs)
            return f_combined, jacs_combined

        return cls(fun_combined_constraint, constraint_list[0].n_state, m_constraint_combined,
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

    def export_scipy(self):
        scifun, scijac = scipinize(self.fun)
        return scifun, scijac


class StepConstraint(Constraint):
    def __init__(self, joint_angle_init, radius):
        n_state = len(joint_angle_init)
        m_constraint = 1

        def fun(joint_angles):
            diff = joint_angles - joint_angle_init
            sqdist = np.linalg.norm(diff)**2 - radius**2
            grad = 2 * diff
            return -np.array([sqdist]), -np.array([grad])

        is_equality = False
        super(StepConstraint, self).__init__(
                fun, n_state, m_constraint, is_equality, with_check=True)

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
            print("debug pose")
            print(diff)
            return diff, J

        super(PoseConstraint, self).__init__(
                fun, n_state, m_constraint, True, with_check=True)

class CollisionConstraint(Constraint):
    def __init__(self, mechanism, static_sdf, with_base=False):
        n_state = mechanism.dof + with_base * 3
        m_constraint = len(mechanism.sphere_ids)

        def fun(joint_angles):
            P0, J = mechanism.collision_forward_kinematics(joint_angles,
                    with_base=with_base, with_jacobian=True)
            J_reshape = J.reshape((m_constraint, 3, n_state))

            sd_vals0 = static_sdf(P0)
            
            sd_jac = np.zeros((m_constraint, 3)) 
            eps = 1e-7
            for i in range(3):
                P1 = copy.copy(P0)
                P1[:, i] += eps
                sd_vals1 = static_sdf(P1)
                sd_jac[:, i] = (sd_vals1 - sd_vals0)/eps

            # sd_jac : (m_constraint, 3)
            # J_reshape : (m_constraint, 3, n_state)
            jac_whole = np.einsum('ij,ijk->ik', sd_jac, J_reshape)

            print("debug col======")
            print(sd_vals0)
            print(mechanism.radius_list)
            sd_vals_sphere = sd_vals0 - np.array(mechanism.radius_list)
            return sd_vals_sphere, jac_whole

        is_equality = False
        super(CollisionConstraint, self).__init__(
                fun, n_state, m_constraint, is_equality, with_check=True)

class Attractor(object):
    def __init__(self, objective_function, ineq_constraint):
        self.objective_function = objective_function
        self.ineq_constraint = ineq_constraint

    def propagate(self, joint_angles, maxiter=50, radius=0.1):
        slsqp_option = {'ftol': 1e-6, 'disp': True, 'maxiter': maxiter}

        step_const = StepConstraint(joint_angles, radius)
        ineq_whole = Constraint.from_constraint_list([step_const, self.ineq_constraint])
        ineq_dict = ineq_whole.export_scipy()

        scifun, scijac = self.objective_function.export_scipy()
        res = scipy.optimize.minimize(
            scifun, joint_angles, method='SLSQP', jac=scijac,
            constraints=[ineq_dict],
            options=slsqp_option, 
            )
        joint_angles_next = res.x
        return joint_angles_next
