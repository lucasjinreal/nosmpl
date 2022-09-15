from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
from .filters import OneEuroFilter
import time



def smooth_quat(quats):
    """
    quats are [129, 24, 4]
    """
    # rots = R.random(23, random_state=233)
    # kt = range(23)

    # downsample 1/3
    quats_downsampled = quats[::3, :, :]
    print(f"down shape: {quats_downsampled.shape}")

    rots = R.from_quat(quats_downsampled)
    kt = range(quats_downsampled.shape[0])

    slerp = Slerp(kt, rots)
    kt_iterp = np.arange(0, quats_downsampled.shape[0] - 1, 0.4)
    print(len(kt_iterp))
    iterp_rots = slerp(kt_iterp)

    quats_interped = iterp_rots.as_quat()

    print(quats_interped.shape)
    return quats_interped


def get_batch_rotmat_from_quat(quats):
    nf = quats.shape[0]
    rots = []
    for i in range(nf):
        rq = R.from_quat(quats[i])
        a = rq.as_matrix()
        rots.append(a)
    rots = np.array(rots)
    return rots


def get_batch_quat_from_rotmat(rots):
    nf = rots.shape[0]
    quats = []
    for i in range(nf):
        rq = R.from_matrix(rots[i])
        a = rq.as_quat()
        quats.append(a)
    quats = np.array(quats)
    return quats


def smooth_oneeural(quats, trans):
    """
    example usage:

    quats = aaa["smpl_pose_quat_wroot"]
    trans = aaa["root_trans"]
    """
    t0 = time.time()
    pred_pose = get_batch_rotmat_from_quat(quats)
    print(pred_pose.shape)

    print(time.time() - t0)

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=0.004,
        beta=0.7,
    )

    new_pose = []
    for idx, pose in enumerate(pred_pose):
        idx += 1
        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        new_pose.append(pose)

    new_pose = np.array(new_pose)
    print(new_pose.shape)
    print(time.time() - t0)

    quats_new = get_batch_quat_from_rotmat(new_pose)

    of_t = OneEuroFilter(
        np.zeros_like(trans[0]),
        trans[0],
        min_cutoff=0.004,
        beta=0.7,
    )
    new_trans = []
    for idx, tr in enumerate(trans):
        idx += 1
        t = np.ones_like(tr) * idx
        n_tr = of_t(t, tr)
        new_trans.append(n_tr)

    new_trans = np.array(new_trans)
    return quats_new, new_trans
