# NoSMPL

An enchanced and accelerated SMPL operation which commonly used in 3D human mesh generation. It takes a poses, shapes, cam_trans as inputs, outputs a high-dimensional 3D mesh verts.

However, SMPL codes and models are so messy out there, they have a lot of codes do calculation, some of them can not be easily deployed or accerlarated. So we have `nosmpl` here, it provides:

- build on smplx, but with onnx support;
- can be inference via onnx;
- we also demantrated some using scenarios infer with `nosmpl` but without any model, only onnx.

This packages provides:

- [ ] Highly optimized pytorch acceleration with FP16 infer enabled;
- [x] Supported ONNX export and infer via ort, so that it might able used into TensorRT or OpenVINO on cpu;
- [x] Support STAR, next generation of SMPL.
- [x] Provide commonly used geoemtry built-in support without torchgeometry or kornia.


STAR model download from: https://star.is.tue.mpg.de/downloads


## Examples

Some pipelines build with `nosmpl` support.

![](https://s4.ax1x.com/2022/02/20/HLGD00.gif)

## Copyrights

Copyrights belongs to Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) and Lucas Jin