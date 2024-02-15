import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from lsq_solver.auto_diff import diff_3point

np.random.seed(2)

rvec_gt = np.random.random(3)
rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()
p3d = np.random.random((3, 2))
rotated_3d_points = rmat_gt @ p3d


def cost_function(rvec: np.ndarray):
    rotation_matrix = Rotation.from_rotvec(rvec).as_matrix()
    return ((rotation_matrix @ p3d) - rotated_3d_points).T.flatten()


def hat(rvec):
    return np.array([[0, -rvec[2], rvec[1]],
                     [rvec[2], 0, -rvec[0]],
                     [-rvec[1], rvec[0], 0]])


def hat_multi(rvec: np.ndarray):
    _, dimension = rvec.reshape(3, -1).shape
    h = np.zeros((3, 3, dimension))
    h[2, 1] = rvec[0]
    h[0, 2] = rvec[1]
    h[1, 0] = rvec[2]
    h[1, 2] = -rvec[0]
    h[2, 0] = -rvec[1]
    h[0, 1] = -rvec[2]
    hts = np.dsplit(h, dimension)
    return np.vstack(hts).squeeze(axis=2)


def rodrigues(rvec):
    rhat = hat(rvec)
    d = np.linalg.norm(rvec)
    return np.eye(3, 3) + np.sin(d) / d * rhat + (1 - np.cos(d)) / d / d * (rhat @ rhat)


def jacobian_l(rvec):
    rhat = hat(rvec)
    d = np.linalg.norm(rvec)
    return np.eye(3, 3) + (1.0 - np.cos(d)) / d / d * rhat + (d - np.sin(d)) / d / d / d * (rhat @ rhat)


if __name__ == '__main__':

    rvec_noise = rvec_gt + np.random.random(3) / 10.0
    print("init:", rvec_noise)

    for iter in range(10):
        print(f"{iter=}")
        residual = cost_function(rvec_noise)

        Jx_numerical = diff_3point((residual.size, 3), cost_function, rvec_noise)
        print(f"Jx numerical: {Jx_numerical.shape}\n{Jx_numerical}")
        rmat_noise = Rotation.from_rotvec(rvec_noise).as_matrix()

        Rs = rmat_noise @ p3d
        Jx_analytic = -hat_multi(Rs) @ jacobian_l(rvec_noise)
        print(f"Jx analytic: {Jx_analytic.shape}\n{Jx_analytic}")

        H = Jx_numerical.T @ Jx_numerical
        B = -Jx_numerical.T @ residual
        dx = scipy.linalg.solve(H, B)
        if (np.linalg.norm(dx) < 1e-8):
            break
        rvec_noise += dx
        print()
    print("end")
    print(f"ground truth:     {rvec_gt}")
    print(f"optimized result: {rvec_noise}")
