JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',  # 0,1,2
    'OP RElbow', 'OP RWrist', 'OP LShoulder',  # 3,4,5
    'OP LElbow', 'OP LWrist', 'OP MidHip',  # 6, 7,8
    'OP RHip', 'OP RKnee', 'OP RAnkle',  # 9,10,11
    'OP LHip', 'OP LKnee', 'OP LAnkle',  # 12,13,14
    'OP REye', 'OP LEye', 'OP REar',  # 15,16,17
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',  # 18,19,20
    # 21, 22, 23, 24  ##Total 25 joints  for openpose
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',  # 0,1,2
    'Left Hip', 'Left Knee', 'Left Ankle',  # 3, 4, 5
    'Right Wrist', 'Right Elbow', 'Right Shoulder',  # 6
    'Left Shoulder', 'Left Elbow', 'Left Wrist',  # 9
    'Neck (LSP)', 'Top of Head (LSP)',  # 12, 13
    'Pelvis (MPII)', 'Thorax (MPII)',  # 14, 15
    'Spine (H36M)', 'Jaw (H36M)',  # 16, 17
    'Head (H36M)', 'Nose', 'Left Eye',  # 18, 19, 20
    'Right Eye', 'Left Ear', 'Right Ear'  # 21,22,23 (Total 24 joints)
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
