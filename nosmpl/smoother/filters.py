import math
import numpy as np


import warnings
import scipy.signal as signal

import numpy as np
import scipy.signal as signal
import torch
import time


class SAVGOLFilter:
    """savgol_filter lib is from:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.savgol_filter.html.

    Args:
        window_size (float):
                    The length of the filter window
                    (i.e., the number of coefficients).
                    window_length must be a positive odd integer.
        polyorder (int):
                    The order of the polynomial used to fit the samples.
                    polyorder must be less than window_length.

    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """

    def __init__(self, window_size=32, polyorder=2):
        super(SAVGOLFilter, self).__init__()
        # bigger polyorder, more same as original poly
        # 1-D Savitzky-Golay filter
        self.window_size = window_size
        self.polyorder = polyorder

    def __call__(self, x=None):
        # x.shape: [t,k,c]
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if window_size <= self.polyorder:
            polyorder = window_size - 1
        else:
            polyorder = self.polyorder
        assert polyorder > 0
        assert window_size > polyorder
        if len(x.shape) != 3:
            warnings.warn("x should be a tensor or numpy of [T*M,K,C]")
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        smooth_poses = np.zeros_like(x)
        # smooth at different axis
        C = x.shape[-1]
        start = time.time()
        for i in range(C):
            smooth_poses[..., i] = signal.savgol_filter(
                x[..., i], window_size, polyorder, axis=0
            )
        end = time.time()
        inference_time = end - start

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)
        return smooth_poses, inference_time


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
