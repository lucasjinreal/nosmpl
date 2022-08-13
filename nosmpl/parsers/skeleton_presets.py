"""

Stores some standard skeleton such as 

CMU
Mixamo

"""

# current only support 25 keypoints
mixamo_corps_name_25 = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToe_End",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToe_End",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]


mixamo_contact_name = [
    "LeftToe_End",
    "RightToe_End",
    "LeftToeBase",
    "RightToeBase",
]


class SkeletonPresets:
    names = [
        "Mixamo",
    ]
    corps_names = [mixamo_corps_name_25]
    contact_names = [
        mixamo_contact_name,
    ]
    contact_thresholds = [0.018]

    @classmethod
    def match(cls, joint_names):
        n_match = []
        for idx, class_name in enumerate(cls.names):
            res = 0
            for j in cls.corps_names[idx]:
                if j in joint_names:
                    res += 1
            n_match.append(res)
        max_match = max(n_match)
        max_match_id = n_match.index(max_match)
        return max_match_id, max_match
