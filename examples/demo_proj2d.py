'''
Project 3d vertices on image?
'''
import sys
import os
import json
import collections
import numpy as np
from alfred import print_shape
from nosmpl.smpl_onnx import SMPLOnnxRuntime, SMPLHOnnxRuntime
from nosmpl.utils import rotmat_to_rotvec

import open3d
import open3d.visualization.rendering as rendering


if __name__ == '__main__':
    model = SMPLHOnnxRuntime()

    data_f = sys.argv[1]
    data = json.load(open(data_f, "r"))
    data = dict((int(key), value) for (key, value) in data.items())
    data = collections.OrderedDict(sorted(data.items()))
    ks = list(data.keys())


    img_width = 512
    img_height = 512
    render = rendering.OffscreenRenderer(img_width, img_height)

    # setup camera intrinsic values
    pinhole = open3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
        
    # Pick a background colour of the rendered image, I set it as black (default is light gray)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

    # now create your mesh
    mesh = open3d.geometry.TriangleMesh()
    mesh.paint_uniform_color([1.0, 0.0, 0.0]) # set Red color for mesh 
    # define further mesh properties, shape, vertices etc  (omitted here)  

    # Define a simple unlit Material.
    # (The base color does not replace the mesh's own colors.)
    mtl = o3d.visualization.rendering.Material()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # add mesh to the scene
    render.scene.add_geometry("MyMeshModel", mesh, mtl)

    # render the scene with respect to the camera
    render.scene.camera.set_projection(camMat, 0.1, 1.0, 640, 480)

    print("frames: ", len(ks), ks[:16])
    for fn in ks:
        data_frame = data[fn]
        pose = data_frame["pose"]
        pose = np.array(pose).reshape(-1, 3, 3)
        print(pose.shape)

        trans = data_frame["trans"]

        pose_rotvec = [rotmat_to_rotvec(i) for i in pose]
        pose_rotvec = np.array(pose_rotvec).reshape(1, 156)
        print(pose_rotvec.shape)

        global_orient = pose_rotvec[:, :3].astype(np.float32)
        # global_orient = [[i[0], -i[1], i[2]] for i in global_orient]
        # global_orient = np.array(global_orient).astype(np.float32)
        body = pose_rotvec[:, 3:66].astype(np.float32)
        lhand = pose_rotvec[:, 66:111].astype(np.float32)
        rhand = pose_rotvec[:, 111:].astype(np.float32)

        print_shape(body, lhand, rhand)

        outputs =model.forward(body, global_orient, lhand, rhand)
           
        vertices, joints, faces = outputs
        vertices = vertices[0].squeeze()
        joints = joints[0].squeeze()

        faces = faces.astype(np.int32)
        # vis_mesh_o3d(vertices, faces)
        # vertices += trans
        # trans = [trans[1], trans[0], trans[2]]

        # now we project the mesh on image?

        