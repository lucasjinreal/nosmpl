"""
utils on visualize SMPL mesh in Open3D

"""
import time
import numpy as np
import os

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    Vector3dVector = o3d.utility.Vector3dVector
    Vector3iVector = o3d.utility.Vector3iVector
    Vector2iVector = o3d.utility.Vector2iVector
    TriangleMesh = o3d.geometry.TriangleMesh
except Exception as e:
    print(e)
    print("run pip install open3d for vis.")
    o3d = None


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        colors = np.array(colors)
        # mesh.vertex_colors = Vector3dVector(colors)
        mesh.paint_uniform_color(colors)
    else:
        r_c = np.random.random(3)
        mesh.paint_uniform_color(r_c)
    return mesh


def vis_mesh_o3d(vertices, faces):

    mesh = create_mesh(vertices, faces)
    min_y = -mesh.get_min_bound()[1]
    mesh.translate([0, min_y, 0])
    o3d.visualization.draw_geometries([mesh])


def vis_mesh_o3d_loop(vertices, faces):

    mesh = create_mesh(vertices, faces)
    min_y = -mesh.get_min_bound()[1]
    mesh.translate([0, min_y, 0])
    o3d.visualization.draw_geometries([mesh])


class Open3DVisualizer:
    def __init__(self, save_img_folder=None, fps=35, enable_axis=False) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("NoSMPL Open3D Visualizer")

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if enable_axis:
            self.vis.add_geometry(coordinate_frame)

        self.geometry_crt = None
        self.fps = fps
        self.idx = 0

        self.save_img_folder = save_img_folder
        if save_img_folder:
            os.makedirs(self.save_img_folder, exist_ok=True)

    def update(self, vertices, faces, trans=None):
        mesh = create_mesh(
            vertices, faces, colors=[82.0 / 255, 217.0 / 255, 118.0 / 255]
        )
        # if not self.geometry_crt:
        #     self.geometry_crt = mesh

        # min_y = -mesh.get_min_bound()[1]
        # mesh.translate([0, min_y, 0])

        if trans:
            mesh.translate(trans)

        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)
        # self.vis.update_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.save_img_folder:
            self.vis.capture_screen_image(
                os.path.join(self.save_img_folder, "temp_%04d.png" % self.idx)
            )
        self.idx += 1
        time.sleep(1/self.fps)
