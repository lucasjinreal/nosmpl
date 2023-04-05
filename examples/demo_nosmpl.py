'''
Demo code to run NoSMPL visualize
'''
from nosmpl.smpl_onnx import SMPLOnnxRuntime
import numpy as np


smpl = SMPLOnnxRuntime()

body = np.random.randn(1, 23, 3).astype(np.float32)
global_orient = np.random.randn(1, 1, 3).astype(np.float32)
outputs = smpl.forward(body, global_orient)
print(outputs)
# you can visualize the verts with Open3D now.