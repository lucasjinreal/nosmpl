import numpy as np
import os
import sys
import torch
import smplx
from nosmpl.vis.vis_o3d import vis_mesh_o3d
from alfred import print_shape, logger
from nosmpl.body_models import SMPLH, SMPL

'''
Usage:

You have to change the model paths to your own
'''

# model_type = "smplh"
model_type = "smpl"

if model_type == "smpl":
    model = SMPL(
        os.path.expanduser(
            "E:\\SMPLs\\SMPL_python_v.1.1.0\\smpl\\models\\SMPL_FEMALE.pkl"
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
    # model = smplx.create(
    #     os.path.expanduser("E:\\SMPLs\\SMPLH_converted\\SMPLH_female.pkl"),
    #     model_type="smplx",
    # )
    model = SMPLH(
        os.path.expanduser("E:\\SMPLs\\SMPLH_converted\\SMPLH_female.pkl"),
        use_pca=False,
    )

num_poses = 4

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

    torch.onnx.export(
        model,
        (None, global_orient, body_pose),
        "smpl.onnx",
        input_names=["global_orient", "body"],
        output_names=["vertices", "joints", "faces"],
    )
    logger.info("exported smpl to onnx.")

    vertices, joints = output
    vertices = vertices.detach().cpu().numpy().squeeze()
    joints = joints.detach().cpu().numpy().squeeze()

    faces = model.faces.astype(np.int32)
    vis_mesh_o3d(vertices, faces)

elif model_type == "smplx":
    betas = torch.randn([1, model.num_betas], dtype=torch.float32).clamp(0, 0.1)
    expression = torch.randn(
        [1, model.num_expression_coeffs], dtype=torch.float32
    ).clamp(0, 0.1)
    body_pose = torch.randn([1, 21, 3], dtype=torch.float32).clamp(0, 0.4)
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

    logger.info(f"betas: {model.num_betas}, expressions: {model.num_expression_coeffs}")

    global_orient = torch.randn([1, 3], dtype=torch.float32).clamp(0, 0.4)
    body_pose = torch.randn([1, 63], dtype=torch.float32).clamp(0, 0.4)
    left_hand_pose = torch.randn([1, 45], dtype=torch.float32).clamp(0, 0.4)
    right_hand_pose = torch.randn([1, 45], dtype=torch.float32).clamp(0, 0.4)
    output = model(
        betas=betas,
        # expression=expression,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=global_orient,
        return_verts=True,
    )
    print(output[0].shape)

    torch.onnx.export(
        model,
        (None, global_orient, body_pose, left_hand_pose, right_hand_pose),
        "smplh.onnx",
        input_names=["global_orient", "body", "lhand", "rhand"],
        output_names=["vertices", "joints", "faces"],
    )
    logger.info("exported smplh to onnx.")

    for i in range(num_poses):
        body_pose = torch.randn([1, 63], dtype=torch.float32).clamp(0, 0.4)
        global_orient = torch.randn([1, 3], dtype=torch.float32)
        print_shape(body_pose, global_orient)
        output = model(
            betas=betas,
            # expression=expression,
            body_pose=body_pose,
            # global_orient=global_orient,
            return_verts=True,
        )

        vertices, joints = output
        vertices = vertices[0].detach().cpu().numpy().squeeze()
        joints = joints[0].detach().cpu().numpy().squeeze()

        faces = model.faces.astype(np.int32)
        vis_mesh_o3d(vertices, faces)
