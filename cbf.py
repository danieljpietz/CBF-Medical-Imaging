import quadprog as qp
import numpy as np


def interp_eval2d(array, p):
    p_floor = np.array([int(np.floor(p[0])), int(np.floor(p[1]))])
    p_ceil = np.array([int(np.ceil(p[0])), int(np.ceil(p[1]))])
    if np.any(p_ceil - p_floor):
        d = np.linalg.norm(p[0:2] - p_floor) / np.linalg.norm(p_ceil - p_floor)
    else:
        d = 0
    return array[tuple(p_floor)] * (1 - d) + array[tuple(p_ceil)] * d


def qpsolve(P, q, G, h):
    qp_G = 0.5 * (P + P.T)
    qp_a = -q
    qp_C = -G.T
    qp_b = -h
    meq = 0
    return qp.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def cbf_eval(syspacket, cbf, uref, lam=(1, 1), umin=None, umax=None):

    (
        dof,
        f,
        j_f,
        g,
    ) = syspacket

    barrier, barrier_gradient, barrier_hessian, dot_barrier = cbf

    ulen = len(uref)

    B1, B2 = lam

    lhs_mult_cbf = barrier_hessian @ f + barrier_gradient @ j_f

    A = np.zeros((1, ulen))
    A[0, :ulen] = (barrier_gradient + lhs_mult_cbf) @ (-g)
    b = np.zeros((1))

    b[0] = (lhs_mult_cbf + barrier_gradient) @ f + B1 * barrier + B2 * dot_barrier

    if umax is not None:
        A = np.concatenate((A, np.eye(dof)))
        b = np.concatenate((b, umax))

    if umin is not None:
        A = np.concatenate((A, -np.eye(dof)))
        b = np.concatenate((b, -umin))

    H = np.eye(ulen)
    f_ = -H @ uref
    u = qpsolve(H, f_, A, b)

    return u[:ulen]
