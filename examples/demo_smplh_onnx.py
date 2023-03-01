"""
Load SMPL-H onnx model

from a pose

generate a mesh using onnxruntime
"""


import collections
import onnxruntime as rt
import torch
import numpy as np
from nosmpl.vis.vis_o3d import vis_mesh_o3d, Open3DVisualizer
import json
from alfred import print_shape
from nosmpl.utils import rot_mat_to_euler


def gen():
    sess = rt.InferenceSession("smplh_sim.onnx")

    for i in range(5):
        body_pose = (
            torch.randn([1, 63], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )
        left_hand_pose = (
            torch.randn([1, 45], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )
        right_hand_pose = (
            torch.randn([1, 45], dtype=torch.float32).clamp(0, 0.4).cpu().numpy()
        )

        outputs = sess.run(
            None, {"body": body_pose, "lhand": left_hand_pose, "rhand": right_hand_pose}
        )

        vertices, joints, faces = outputs
        vertices = vertices[0].squeeze()
        joints = joints[0].squeeze()

        faces = faces.astype(np.int32)
        vis_mesh_o3d(vertices, faces)


def vis_json():
    sess = rt.InferenceSession("smplh_sim.onnx")

    data = json.load(open("test.json", "r"))
    data = dict((int(key), value) for (key, value) in data.items())
    data = collections.OrderedDict(sorted(data.items()))
    ks = list(data.keys())

    o3d_vis = Open3DVisualizer(fps=60, enable_axis=False)

    print("frames: ", len(ks), ks[:16])
    for fn in ks:
        data_frame = data[fn]
        pose = data_frame["pose"]
        pose = np.array(pose).reshape(-1, 3, 3)
        print(pose.shape)

        trans = data_frame["trans"]

        pose_euler = [rot_mat_to_euler(i) for i in pose]
        pose_euler = np.array(pose_euler).reshape(1, 156)
        print(pose_euler.shape)

        body = pose_euler[:, 3:66].astype(np.float32)
        lhand = pose_euler[:, 66:111].astype(np.float32)
        rhand = pose_euler[:, 111:].astype(np.float32)

        print_shape(body, lhand, rhand)

        outputs = sess.run(None, {"body": body, "lhand": lhand, "rhand": rhand})

        vertices, joints, faces = outputs
        vertices = vertices[0].squeeze()
        joints = joints[0].squeeze()

        faces = faces.astype(np.int32)
        # vis_mesh_o3d(vertices, faces)
        # vertices += trans
        # trans = [trans[1], trans[0], trans[2]]
        trans = [trans[0], trans[1], 0]
        print(trans)
        o3d_vis.update(vertices, faces, trans)


if __name__ == "__main__":
    # gen()
    vis_json()
