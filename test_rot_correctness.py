'''
Test rotation conversion in nosmpl correctness

we compare with Scipy && Blender

from:

aa -> quaternion
rotmat -> quaternion
'''
from re import A
from nosmpl.geometry import aa2quat
from scipy.spatial.transform import Rotation as R
import numpy as np


if __name__ == '__main__':
    
    a = np.random.randn([24, 3, 3])
    aa = R.from_matrix(a)

    aaa = aa.as_quat()
    print(aaa)
