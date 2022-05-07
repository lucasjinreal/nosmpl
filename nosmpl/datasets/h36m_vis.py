from __future__ import division
from bitarray import test

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import copy
from . import h36m_data_utils as data_utils
import os


class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = (
            np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        )
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots.append(
                self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor)
            )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax._axis3don = True  # this line use to hide the 3D frame

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Update the plotted 3d pose.
        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert channels.size == 96, (
            "channels should have 96 entries, it has %d instead" % channels.size
        )
        vals = np.reshape(channels, (32, -1))

        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

        r = 750
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

        self.ax._axis3don = True
        self.ax.set_aspect("auto")


def fkl(angles, parent, offset, rotInd, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    Based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if not rotInd[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]

        thisRotation = data_utils.expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:  # Root node
            xyzStruct[i]["rotation"] = thisRotation
            xyzStruct[i]["xyz"] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]["xyz"] = (offset[i, :] + thisPosition).dot(
                xyzStruct[parent[i]]["rotation"]
            ) + xyzStruct[parent[i]]["xyz"]
            xyzStruct[i]["rotation"] = thisRotation.dot(
                xyzStruct[parent[i]]["rotation"]
            )

    xyz = [xyzStruct[i]["xyz"] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    xyz = xyz[:, [0, 2, 1]]
    # print(xyz)
    return np.reshape(xyz, [-1])


def revert_coordinate_space(channels, R0, T0):
    """
    Bring a series of poses to a canonical form so they are facing the camera when they start.
    Adapted from
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

    Args
      channels: n-by-99 matrix of poses
      R0: 3x3 rotation for the first frame
      T0: 1x3 position for the first frame
    Returns
      channels_rec: The passed poses, but the first has T0 and R0, and the
                    rest of the sequence is modified accordingly.
    """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):
        R_diff = data_utils.expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))).reshape(-1)
        channels_rec[ii, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def _some_variables():
    """
    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = (
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                7,
                8,
                9,
                10,
                1,
                12,
                13,
                14,
                15,
                13,
                17,
                18,
                19,
                20,
                21,
                20,
                23,
                13,
                25,
                26,
                27,
                28,
                29,
                28,
                31,
            ]
        )
        - 1
    )

    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )
    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)
    return parent, offset, rotInd, expmapInd


def h36m_load_gt_3d_data(gt_file):

    """
    returns the 3d coordinates which can be shown rather than angle-axis data
    which also contains converted rotation between each frame.
    """
    parent, offset, rotInd, expmapInd = _some_variables()

    test_set, _ = data_utils.load_data_from_file(gt_file)
    expmap_gt = test_set
    nframes_gt = expmap_gt.shape[0]
    expmap_gt = expmap_gt[:nframes_gt, :]

    # Compute 3d points for each frame
    xyz_gt = np.zeros((nframes_gt, 96))
    for i in range(nframes_gt):
        xyz_gt[i, :] = fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd)
    return xyz_gt


def h36m_vis_on_gt_file(gt_file):
    parent, offset, rotInd, expmapInd = _some_variables()

    test_set, _ = data_utils.load_data_from_file(gt_file)
    expmap_gt = test_set
    expmap_pred = np.zeros_like(test_set)

    nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

    # Put them together and revert the coordinate space
    expmap_all = revert_coordinate_space(
        np.vstack((expmap_gt, expmap_pred)), np.eye(3), np.zeros(3)
    )
    expmap_gt = expmap_all[:nframes_gt, :]
    expmap_pred = expmap_all[nframes_gt:, :]

    # Compute 3d points for each frame
    xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
    for i in range(nframes_gt):
        xyz_gt[i, :] = fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd)
    for i in range(nframes_pred):
        xyz_pred[i, :] = fkl(expmap_pred[i, :], parent, offset, rotInd, expmapInd)

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.gca(projection="3d")
    ob = Ax3DPose(ax)

    to_draw = np.append(xyz_gt, xyz_pred, axis=0)

    # dirty workround for generation gif
    counter = 0

    def update(x):
        nonlocal counter
        if counter < 25:
            counter += 1
            return ob.update(x)
        else:
            if counter == 50:
                counter = 0
            else:
                counter += 1
            return ob.update(x, lcolor="#9b59b6", rcolor="#2ecc71")

    anim = animation.FuncAnimation(fig, update, frames=to_draw, interval=40)
    plt.show()
