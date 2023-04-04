import torch
import torch.nn.functional as F
from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

Tensor = NewType("Tensor", torch.Tensor)
Array = NewType("Array", np.ndarray)


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(array: Union[Array, Tensor], dtype=torch.float32) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]
    if isinstance(rot_mats, torch.Tensor):
        sy = torch.sqrt(
            rot_mats[:, 0, 0] * rot_mats[:, 0, 0]
            + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
        )
        return torch.atan2(-rot_mats[:, 2, 0], sy)
    else:
        assert len(rot_mats.shape) == 2, "numpy mode support only 3x3"

        # r = R.from_matrix(rot_mats)
        # res = r.as_euler('xyz', degrees=False)
        # return res

        # res = np.array(
        #     [
        #         np.arctan2(rot_mats[2, 1], rot_mats[2, 2]),
        #         np.arctan2(
        #             -rot_mats[2, 0], np.sqrt(rot_mats[2, 1] ** 2 + rot_mats[2, 2] ** 2)
        #         ),
        #         np.arctan2(rot_mats[1, 0], rot_mats[0, 0]),
        #     ]
        # )
        R = rot_mats
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])


def rotmat_to_rotvec(rot_mats):
    assert isinstance(rot_mats, np.ndarray), "only accept numpy for now"
    r = R.from_matrix(rot_mats)
    res = r.as_rotvec()
    return res


def quat_to_rotvec(quat):
    r = R.from_quat(quat)
    return r.as_rotvec()

def quat_feat(theta):
    """
        Computes a normalized quaternion ([0,0,0,0]  when the body is in rest pose)
        given joint angles
    :param theta: A tensor of joints axis angles, batch size x number of joints x 3
    :return:
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized, v_cos - 1], dim=1)
    return quat


def quat2mat(quat):
    """
        Converts a quaternion to a rotation matrix
    :param quat:
    :return:
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    """
        Computes the rodrigues representation given joint angles

    :param theta: batch_size x number of joints x 3
    :return: batch_size x number of joints x 3 x 4
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


def with_zeros(input):
    """
      Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor

    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    """
    batch_size = input.shape[0]
    row_append = torch.cuda.FloatTensor(([0.0, 0.0, 0.0, 1.0]))
    row_append.requires_grad = False
    padded_tensor = torch.cat(
        [input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1
    )
    return padded_tensor


"""
For SMPL
"""


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_global_rigid_transformation(Rs, Js, parent):

    # Now Js is N x 24 x 3 x 1
    Js = torch.unsqueeze(Js, -1)
    rel_joints = Js.clone()
    rel_joints[:, 1:] -= Js[:, parent[1:]]

    transforms_mat = transform_mat(
        Rs.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, Js.shape[1], 4, 4)

    results = [transforms_mat[:, 0]]
    for i in range(1, parent.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        res_here = torch.matmul(results[parent[i]], transforms_mat[:, i])
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = F.pad(Js, [0, 0, 0, 1])
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = F.pad(init_bone, (3, 0), "constant", 0)
    A = results - init_bone

    return new_J, A


def batch_rodrigues(theta, return_quat=False):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    if return_quat:
        return quat
    return quat2mat(quat)
