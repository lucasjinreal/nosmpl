import onnxruntime as rt
from alfred.utils.file_io import download
import os
from alfred import logger


'''
this 2 class helps you get SMPL verts via onnx
the onnx model download automatically
'''


class SMPLOnnxRuntime:

    def __init__(self) -> None:
        preset_file_path = os.path.join(
            os.path.expanduser('~'), 'cache/nosmpl')
        os.makedirs(preset_file_path, exist_ok=True)
        tgt_f = os.path.join(preset_file_path, 'smpl.onnx')
        if not os.path.exists(tgt_f):
            download('https://github.com/jinfagang/nosmpl/releases/download/v1.1/smpl.onnx',
                     preset_file_path)
        self.sess = rt.InferenceSession(tgt_f)
        logger.info('SMPL onnx loaded!')

    def forward(self, body_pose_rotvec, global_orient):
        '''
        outputs: vertices, joints, faces
        '''
        outputs = self.sess.run(
            None,
            {
                "global_orient": global_orient,
                "body": body_pose_rotvec,
            },
        )
        return outputs


class SMPLHOnnxRuntime:

    def __init__(self) -> None:
        preset_file_path = os.path.join(
            os.path.expanduser('~'), 'cache/nosmpl')
        os.makedirs(preset_file_path, exist_ok=True)
        tgt_f = os.path.join(preset_file_path, 'smplh.onnx')
        if not os.path.exists(tgt_f):
            download('https://github.com/jinfagang/nosmpl/releases/download/v1.0/smplh_sim_w_orien.onnx',
                     preset_file_path)
        self.sess = rt.InferenceSession(tgt_f)
        logger.info('SMPL-H onnx loaded!')

    def forward(self, body_pose_rotvec, global_orient, lhand, rhand):
        '''
        outputs: vertices, joints, faces
        '''
        outputs = self.sess.run(
            None,
            {
                "global_orient": global_orient,
                "body": body_pose_rotvec,
                "lhand": lhand,
                "rhand": rhand
            },
        )
        return outputs
