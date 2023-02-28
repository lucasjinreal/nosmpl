import numpy as np
import os
import sys
import torch
import smplx
from nosmpl.vis.vis_o3d import vis_mesh_o3d
from alfred import print_shape, logger
from os import path as osp


model_type = "smplh"

if model_type == "smpl":
    model = smplx.create(
        os.path.expanduser(
            "~/data/face_and_pose/SMPL_python_v.1.1.0/smpl/models/SMPL_FEMALE.pkl"
        ),
        # model_type="smplx",
        model_type="smpl",
    )
elif model_type == "smplx":
    model = smplx.create(
        os.path.expanduser("~/data/face_and_pose/smplx/SMPLX_FEMALE.npz"),
        model_type="smplx",
    )
elif model_type == "smplh":
    model = smplx.create(
        os.path.expanduser("E:\\SMPLs\\SMPLH\\SMPLH_female.pkl"),
        model_type="smplx",
    )


num_poses = 4  # number of body poses in each batch


if model_type == "smpl":
    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)
    body_pose = torch.randn([1, 23, 3], dtype=torch.float32)
    global_orient = torch.randn([1, 1, 3], dtype=torch.float32)
    output = model(
        # betas=betas, expression=expression, body_pose=body_pose, return_verts=True
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        return_verts=True,
    )
elif model_type == "smplx":
    betas = torch.randn([1, model.num_betas], dtype=torch.float32).clamp(0, 0.1)
    expression = torch.randn(
        [1, model.num_expression_coeffs], dtype=torch.float32
    ).clamp(0, 0.1)
    body_pose = torch.randn([1, 21, 3], dtype=torch.float32).clamp(0, 0.4)

    for i in range(num_poses):
        body_pose = None
        output = model(
            betas=betas, expression=expression, body_pose=body_pose, return_verts=True
        )

        vertices = output.vertices[0].detach().cpu().numpy().squeeze()
        joints = output.joints[0].detach().cpu().numpy().squeeze()

        faces = model.faces.astype(np.int32)
        vis_mesh_o3d(vertices, faces)

elif model_type == "smplh":
    betas = torch.randn([1, model.num_betas], dtype=torch.float32).clamp(0, 0.1)
    expression = torch.randn(
        [1, model.num_expression_coeffs], dtype=torch.float32
    ).clamp(0, 0.1)
    body_pose = torch.randn([1, 21, 3], dtype=torch.float32).clamp(0, 0.4)
    logger.info(f'betas: {model.num_betas}, expressions: {model.num_expression_coeffs}')

    for i in range(num_poses):
        body_pose = None
        output = model(
            betas=betas, expression=expression, body_pose=body_pose, return_verts=True
        )

        vertices = output.vertices[0].detach().cpu().numpy().squeeze()
        joints = output.joints[0].detach().cpu().numpy().squeeze()

        faces = model.faces.astype(np.int32)
        vis_mesh_o3d(vertices, faces)
