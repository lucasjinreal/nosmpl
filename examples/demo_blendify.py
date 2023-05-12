# Script to render cube
from blendify import scene
from blendify.materials import PrinsipledBSDFMaterial
from blendify.colors import UniformColors
import os
import sys
import trimesh
import numpy as np


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

if __name__ == '__main__':

    # Add light
    scene.lights.add_point(strength=1000, translation=(4, -2, 4))
    # Add camera
    # scene.set_perspective_camera((512, 512), fov_x=0.7)
    scene.set_perspective_camera(
        # (1024, 1024), fov_x=np.deg2rad(69.1), quaternion=(-0.707, -0.707, 0.0, 0.0),
        (1024, 1024), fov_x=np.deg2rad(69.1), quaternion=(-0.6, 0, 0.0, 0.0),
        translation=[-0.0, -0.0, 0.0]
    )
    

    model_p = os.path.abspath(sys.argv[1])
    print(f"loading model to render: {model_p}")

    mesh = as_mesh(trimesh.load_mesh(model_p, process=False))
    # mesh = scene_model.meshes[0]
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)

    # Create material
    material = PrinsipledBSDFMaterial()
    # Create color
    color = UniformColors((0.0, 1.0, 0.0))
    # Add cube mesh
    scene.renderables.add_mesh(
        vertices=vertices, faces=faces, material=material, colors=color
    )
    # Render scene
    scene.render(filepath=os.path.join(os.path.dirname(__file__), "cube.png"))
