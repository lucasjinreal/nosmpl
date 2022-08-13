import torch
import numpy as np


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_d2 = mask_d2.int()
    mask_d0_d1 = mask_d0_d1.int()
    mask_d0_nd1 = mask_d0_nd1.int()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_angle_axis_torch(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def __rotation_matrix_to_angle_axis_torch(rotmat):
    rotmat = rotmat.clone()
    if rotmat.dim() == 2:
        assert rotmat.size(0) == 3
        assert rotmat.size(1) == 3
        rotmat0 = torch.zeros((1, 3, 4))
        rotmat0[0, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = rotation_matrix_to_angle_axis_torch(rotmat0)[0]
    elif rotmat.dim() == 3:
        dim0 = rotmat.size(0)
        assert rotmat.size(1) == 3
        assert rotmat.size(2) == 3
        rotmat0 = torch.zeros((dim0, 3, 4))
        rotmat0[:, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = rotation_matrix_to_angle_axis_torch(rotmat0)
    else:
        assert rotmat.dim() == 4
        dim0 = rotmat.size(0)
        dim1 = rotmat.size(1)
        assert rotmat.size(2) == 3
        assert rotmat.size(3) == 3
        rotmat0 = torch.zeros((dim0 * dim1, 3, 4))
        rotmat0[:, :, :3] = rotmat.view(dim0 * dim1, 3, 3)
        rotmat0[:, 2, 3] = 1.0
        aa = rotation_matrix_to_angle_axis_torch(rotmat0)
        aa = aa.view(dim0, dim1, 3)
    return aa


def rotation_matrix_to_angle_axis(rotmat):
    # TODO: does torch is really needed here??
    if isinstance(rotmat, torch.Tensor):
        return __rotation_matrix_to_angle_axis_torch(rotmat)
    else:
        assert isinstance(rotmat, np.ndarray)
        rotmat_torch = torch.from_numpy(rotmat)
        aa_torch = __rotation_matrix_to_angle_axis_torch(rotmat_torch)
        return aa_torch.numpy()


def quat2euler(q, order="xyz", degrees=True):
    """
    Convert (w, x, y, z) quaternions to xyz euler angles. This is  used for bvh output.
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.empty(q0.shape + (3,), device=q.device, dtype=q.dtype)

    if order == "xyz":
        es[..., 2] = torch.atan2(
            2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        )
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(
            2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
        )
    else:
        raise NotImplementedError("Cannot convert to ordering %s" % order)

    if degrees:
        es = es * 180 / np.pi
    return es


def quat2mat(quats: torch.Tensor):
    """
    Convert (w, x, y, z) quaternions to 3x3 rotation matrix
    :param quats: quaternions of shape (..., 4)
    :return:  rotation matrices of shape (..., 3, 3)
    """
    qw = quats[..., 0]
    qx = quats[..., 1]
    qy = quats[..., 2]
    qz = quats[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quats.shape[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def quat2repr6d(quat):
    mat = quat2mat(quat)
    res = mat[..., :2, :]
    res = res.reshape(res.shape[:-2] + (6,))
    return res


def mat2quat(R) -> torch.Tensor:
    """
    https://github.com/duolu/pyrotation/blob/master/pyrotation/pyrotation.py
    Convert a rotation matrix to a unit quaternion.

    This uses the Shepperd’s method for numerical stability.
    """

    # The rotation matrix must be orthonormal

    w2 = 1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    x2 = 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]
    y2 = 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]
    z2 = 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]

    yz = R[..., 1, 2] + R[..., 2, 1]
    xz = R[..., 2, 0] + R[..., 0, 2]
    xy = R[..., 0, 1] + R[..., 1, 0]

    wx = R[..., 2, 1] - R[..., 1, 2]
    wy = R[..., 0, 2] - R[..., 2, 0]
    wz = R[..., 1, 0] - R[..., 0, 1]

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    res = [w, x, y, z]
    res = [z.unsqueeze(-1) for z in res]

    return torch.cat(res, dim=-1) / 2


def repr6d2quat(repr) -> torch.Tensor:
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat2quat(mat)


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def euler2mat(rots, order="xyz"):
    axis = {
        "x": torch.tensor((1, 0, 0), device=rots.device),
        "y": torch.tensor((0, 1, 0), device=rots.device),
        "z": torch.tensor((0, 0, 1), device=rots.device),
    }

    rots = rots / 180 * np.pi
    mats = []
    for i in range(3):
        aa = axis[order[i]] * rots[..., i].unsqueeze(-1)
        mats.append(aa2mat(aa))
    return mats[0] @ (mats[1] @ mats[2])


def aa2mat(rots):
    """
    Convert angle-axis representation to rotation matrix
    :param rots: angle-axis representation
    :return:
    """
    quat = aa2quat(rots)
    mat = quat2mat(quat)
    return mat


def aa2quat(rots, form="wxyz", unified_orient=True):
    """
    Convert angle-axis representation to wxyz quaternion and to the half plan (w >= 0)
    @param rots: angle-axis rotations, (*, 3)
    @param form: quaternion format, either 'wxyz' or 'xyzw'
    @param unified_orient: Use unified orientation for quaternion (quaternion is dual cover of SO3)
    :return:
    """
    angles = rots.norm(dim=-1, keepdim=True)
    norm = angles.clone()
    norm[norm < 1e-8] = 1
    axis = rots / norm
    quats = torch.empty(rots.shape[:-1] + (4,), device=rots.device, dtype=rots.dtype)
    angles = angles * 0.5
    if form == "wxyz":
        quats[..., 0] = torch.cos(angles.squeeze(-1))
        quats[..., 1:] = torch.sin(angles) * axis
    elif form == "xyzw":
        quats[..., :3] = torch.sin(angles) * axis
        quats[..., 3] = torch.cos(angles.squeeze(-1))

    if unified_orient:
        idx = quats[..., 0] < 0
        quats[idx, :] *= -1

    return quats
