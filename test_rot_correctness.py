'''
Test rotation conversion in nosmpl correctness

we compare with Scipy && Blender

from:

aa -> quaternion
rotmat -> quaternion
'''

try:
    import bpy
    from mathutils import Matrix, Vector, Quaternion, Euler
    import numpy as np
except ImportError:
    bpy = None
    from nosmpl.geometry import aa2quat
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from nosmpl.geometry import rotation_matrix_to_angle_axis
    # from mmhuman3d.utils.geometry import rotation_matrix_to_angle_axis as rotation_matrix_to_angle_axis_mh
    import torch


def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


if __name__ == '__main__':

    if bpy is None:
        np.random.seed(21023)
        a = np.random.rand(24, 3, 3)
        a, _ = np.linalg.qr(a)
        aa = R.from_matrix(a)

        aaa = aa.as_quat()
        print(aaa)

        b = rotation_matrix_to_angle_axis(a)
        # b2 = rotation_matrix_to_angle_axis_mh(torch.as_tensor(a))
        c = aa2quat(b)
        print(c)
        print('---------')
        # print(b2)
        # print(b)
    else:
        print('runing bpy')
        np.random.seed(21023)
        a = np.random.rand(24, 3, 3)
        a, _ = np.linalg.qr(a)
        # pose = rotation_matrix_to_angle_axis(a)
        mat_rots = a

        res = []
        for mat_rot in mat_rots:
            if len(mat_rot) == 3:
                mat_rot = Matrix(mat_rot).to_quaternion()
            bone_rotation = Quaternion(mat_rot)
            res.append(bone_rotation)
        res = np.asarray(res)
        print(res)
