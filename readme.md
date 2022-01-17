# SMPL2

An enchanced and accelerated SMPL operation which commonly used in 3D human mesh generation. It takes a poses, shapes, cam_trans as inputs, outputs a high-dimensional 3D mesh verts.

This packages provides:

- [ ] Highly optimized pytorch acceleration with FP16 infer enabled;
- [ ] Supported ONNX export and infer via ort, so that it might able used into TensorRT or OpenVINO on cpu;
- [ ] Support STAR, next generation of SMPL.
- [ ] Provide commonly used geoemtry built-in support without torchgeometry or kornia.


## Examples

Some pipelines build with SMPL2 support.



## Copyrights

Copyrights belongs to Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) and Lucas Jin