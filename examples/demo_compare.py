import numpy as np
import time
from torch.autograd import Variable
import torch
from nosmpl.star import STAR
from nosmpl.smpl import SMPL2


star = STAR(model_path='data/star_1_1/female/model.npz')
smpl = SMPL2(model_path='/media/jintian/samsung/source/ai/swarm/toolchains/mmkd/vendor/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl')
smpl.to('cuda')


batch_size = 1

# 姿态参数theta：24x3=72
poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
poses = Variable(poses, requires_grad=True)
# 体型参数beta：10
betas = torch.cuda.FloatTensor(np.zeros((batch_size, 10)))
betas = Variable(betas, requires_grad=True)
# 相机参数trans：3
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
trans = Variable(trans, requires_grad=True)
t0 = time.perf_counter()
d = star(poses, betas, trans)
t1 = time.perf_counter()
print(t1 - t0)

# 生成STAR的obj文件
d_np = d.cpu().detach().numpy()
print(d_np.shape)
outmesh_path = './star.obj'
with open(outmesh_path, 'w') as fp:
    for i in d_np:
        for v in i:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    for f in star.f+1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
print("STAR模型已建立")


poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
poses = Variable(poses, requires_grad=True)
t0 = time.perf_counter()
d = smpl(betas, poses)
t1 = time.perf_counter()
print(t1 - t0)
