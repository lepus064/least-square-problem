import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from lsq_solver.auto_diff import diff_3point

np.random.seed(2)

rvec_gt = np.random.random(3)
rmat_gt = Rotation.from_rotvec(rvec_gt).as_matrix()
p3d = np.random.random((3, 2))


def f(rvec):
    m = Rotation.from_rotvec(rvec).as_matrix()
    return ((rmat_gt @ p3d) - (m @ p3d)).flatten()

def hat(rvec):
    return np.array([[0, -rvec[2], rvec[1]],
                     [rvec[2], 0, -rvec[0]],
                     [-rvec[1], rvec[0], 0]])

def rodrigues(rvec):
    rhat = hat(rvec)
    d = np.linalg.norm(rvec)
    return np.eye(3, 3) + np.sin(d) / d * rhat + (1 - np.cos(d))/ d/ d* (rhat@rhat)

def jacobian_l(rvec):
    rhat = hat(rvec)
    d = np.linalg.norm(rvec)
    return np.eye(3, 3) + (1.0 - np.cos(d))/d/d*rhat + (d - np.sin(d))/d/d/d*(rhat@rhat)

G1 = np.zeros((3, 3))
G1[2, 1] = 1.0
G1[1, 2] = -1.0
G2 = np.zeros((3, 3))
G2[2, 0] = -1.0
G2[0, 2] = 1.0
G3 = np.zeros((3, 3))
G3[1, 0] = 1.0
G3[0, 1] = -1.0

if __name__ == '__main__':

    rvec_noise = rvec_gt + np.random.random(3) / 10.0
    print("init:", rvec_noise)

    for iter in range(10):
        print(iter)
        residual = f(rvec_noise)
        print(f"residual:{residual.shape}")
        
        Jx = diff_3point((residual.size, 3), f, rvec_noise)
        print(f"Jx: {Jx.shape}\n{Jx}")
        rmat_noise = Rotation.from_rotvec(rvec_noise).as_matrix()
        # print("r", rmat_noise)
        # print(rodrigues(rvec_noise))
        # exit()
        Rs0 = rmat_noise@p3d[:,0]
        jj0 = hat(Rs0) @ jacobian_l(rvec_noise)
        Rs1 = rmat_noise@p3d[:,1]
        jj1 = hat(Rs1) @ jacobian_l(rvec_noise)
        j = np.hstack((jj0, jj1)).reshape(-1, 3)
        print(j-Jx)
        # rr = -rmat_noise @ p3d
        # print(rr)

        H = Jx.T @ Jx
        B = -Jx.T @ residual
        dx = scipy.linalg.solve(H, B)
        if(np.linalg.norm(dx) < 1e-8):
            break
        rvec_noise += dx
        print()
    print("end")
    print(rvec_gt)
    print(rvec_noise)

# print("rmat gt:\n", rmat_gt)
# print(j)