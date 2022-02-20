from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_global_rigid_transformation, batch_rodrigues
from .lbs import vertices2joints
from .vertex_joint_selector import VertexJointSelector
from .vertex_ids import vertex_ids


class SMPL(nn.Module):
    def __init__(
        self,
        model_path,
        extra_regressor=None,
        h36m_regressor=None,
        dtype=torch.float32,
    ):
        super(SMPL, self).__init__()
        self.dtype = dtype
        with open(model_path, "rb") as fp:
            smpl_model = pickle.load(fp, encoding="latin1")

        v_template = np.array(smpl_model["v_template"], dtype=np.float)
        v_template = torch.tensor(v_template, dtype=dtype)
        self.register_buffer("v_template", v_template)

        self.size = [self.v_template.size(0), 3]  # [num of vertices, 3]
        self.num_betas = smpl_model["shapedirs"].shape[-1]  # 10

        shapedirs = np.array(smpl_model["shapedirs"])  # 6980 x 3 x 10
        shapedirs = np.reshape(shapedirs, [-1, self.num_betas]).T
        shapedirs = torch.tensor(shapedirs, dtype=dtype)  # 10x6980*3
        self.register_buffer("shapedirs", shapedirs)

        J_regressor = np.array(smpl_model["J_regressor"].T.todense())
        J_regressor = torch.tensor(J_regressor, dtype=dtype)  # 6890 x 24
        self.register_buffer("J_regressor", J_regressor)

        num_pose_basis = smpl_model["posedirs"].shape[-1]
        posedirs = np.array(smpl_model["posedirs"])
        posedirs = np.reshape(posedirs, [-1, num_pose_basis]).T
        posedirs = torch.tensor(posedirs, dtype=dtype)
        self.register_buffer("posedirs", posedirs)

        self.parents = smpl_model["kintree_table"][0].astype(np.int32)

        weights = np.array(smpl_model["weights"])
        weights = torch.tensor(weights, dtype=dtype)
        self.register_buffer("weights", weights)

        if extra_regressor:
            # this is used in SPIN and FrankMocap
            extra_regressor = torch.from_numpy(
                np.load(extra_regressor)).to(self.dtype)
            self.register_buffer("extra_regressor", extra_regressor)

        if h36m_regressor:
            h36m_regressor = torch.from_numpy(
                np.load(h36m_regressor)).to(self.dtype)
            self.register_buffer("h36m_regressor", h36m_regressor)

        self.register_buffer("e3", torch.eye(3))

        # we will produce faces as well (for visualization if needed)
        self.faces = np.array(smpl_model["f"]).astype(np.int32)
        faces_tensor = torch.tensor(self.faces, dtype=torch.int)
        self.register_buffer("faces_tensor", faces_tensor)

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids["smplh"])

        # need so this outside SMPL, we just return them all
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        # # TODO: this is fixed for FrankMocap
        # self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, beta, theta, is_axis_angle=False):
        """
        beta is predicted shapes
        theta is predict rotmat, but if is_axis_angle == True, you need
        convert rotmat to AxisAngle first.

        is_axis_angle = TRUE, theta is axis angle
        is_axis_angle = False, theta is rotmat
        return joints shape: B x 25 x 3
        """
        batch_size = beta.size(0)
        v_shaped = torch.matmul(beta, self.shapedirs)
        v_shaped = v_shaped.view(-1,
                                 self.size[0], self.size[1]) + self.v_template

        J = torch.einsum("bjm,jn->bnm", v_shaped, self.J_regressor)

        if is_axis_angle:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else:
            Rs = theta.view(-1, 24, 3, 3)

        pose_feature = torch.reshape(
            Rs[:, 1:, :, :] - self.e3, (-1, 207)).to(self.dtype)
        v_posed = torch.matmul(
            pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents)

        T = torch.einsum(
            "bnm,vn->bvm", A.view(batch_size, 24, 16), self.weights)
        T = T.view(batch_size, -1, 4, 4)

        v_posed_homo = F.pad(v_posed, (0, 1, 0, 0, 0, 0), "constant", 1)

        v_homo = torch.matmul(
            T,
            v_posed_homo.view(batch_size, v_posed_homo.size(
                1), v_posed_homo.size(2), 1),
        ).to(self.dtype)

        verts = v_homo[:, :, :3, 0]
        joints = self.vertex_joint_selector(verts, J_transformed)

        # returned joints can be 24, 25, or 24 + 21 (if 21, then face and hand feet are applied)
        if self.extra_regressor is not None:
            joints_extra = vertices2joints(self.extra_regressor, verts)
            joints = torch.cat([joints, joints_extra], dim=1)
            # we might have a mapper for all joints
            # joints = joints[:, self.joint_map, :]
            return verts, joints, self.faces_tensor
        else:
            # here, returned 24 + 21
            return verts, joints, self.faces_tensor


def export_smpl_to_onnx(smpl_model, save_file, bs=1):
    os.makedirs(save_file, exist_ok=True)
    a = torch.rand([bs, 10]).to(device)
    b = torch.rand([bs, 24, 3, 3]).to(device)
    torch.onnx.export(smpl_model, (a, b),
                      save_file,
                      output_names=['verts', 'joints', 'faces'],
                      opset_version=12)
    print('SMPL onnx saved into: ', save_file)
