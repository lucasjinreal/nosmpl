from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_global_rigid_transformation, batch_rodrigues
# from .vertices_selector import VertexJointSelector
# from .vertices_ids import vertex_ids
# from .constants import JOINT_MAP, JOINT_NAMES_USED
import time


class SMPL2(nn.Module):
    def __init__(
        self,
        model_path,
        extra_regressor,
        h36m_regressor,
        dtype=torch.float32,
    ):
        super(SMPL2, self).__init__()
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

        extra_regressor = torch.from_numpy(
            np.load(extra_regressor)).to(self.dtype)
        self.register_buffer("extra_regressor", extra_regressor)

        h36m_regressor = torch.from_numpy(
            np.load(h36m_regressor)).to(self.dtype)
        self.register_buffer("h36m_regressor", h36m_regressor)

        self.register_buffer("e3", torch.eye(3))

        # self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids["smplh"])
        # self.joint_mapper = [JOINT_MAP[i] for i in JOINT_NAMES_USED]

    def forward(self, beta, theta, aa=True, return_verts=False):
        """
        aa = TRUE, theta is axis angle
        aa = False, theta is rotmat
        return joints shape: B x 25 x 3
        """
        batch_size = beta.size(0)
        # print(beta.dtype, theta.dtype)
        # t0 = time.perf_counter()
        v_shaped = torch.matmul(beta, self.shapedirs)
        v_shaped = v_shaped.view(-1,
                                 self.size[0], self.size[1]) + self.v_template

        J = torch.einsum("bjm,jn->bnm", v_shaped, self.J_regressor)

        if aa:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else:
            Rs = theta.view(-1, 24, 3, 3)

        pose_feature = torch.reshape(
            Rs[:, 1:, :, :] - self.e3, (-1, 207)).to(self.dtype)
        # print(pose_feature.dtype, self.posedirs.dtype)
        v_posed = torch.matmul(
            pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents)
        t1 = time.perf_counter()

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
        # joints_extra = torch.einsum("bvn,jv->bjn", verts, self.extra_regressor)

        # joints_op = self.vertex_joint_selector(verts, J_transformed)
        # joints = torch.cat([joints_op, joints_extra], dim=1)
        # joints = joints[:, self.joint_mapper]

        # # t2 = time.perf_counter()
        # # print('second: ', t2 - t0)

        # if return_verts:
        #     return verts, joints

        # return joints
        return verts
