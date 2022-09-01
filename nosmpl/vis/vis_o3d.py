"""
utils on visualize SMPL mesh in Open3D

"""
import numpy as np

try:
    import open3d as o3d
    Vector3dVector = o3d.utility.Vector3dVector
    Vector3iVector = o3d.utility.Vector3iVector
    Vector2iVector = o3d.utility.Vector2iVector
    TriangleMesh = o3d.geometry.TriangleMesh
except Exception as e:
    print(e)
    o3d = None


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        r_c = np.random.random(3)
        print(r_c)
        mesh.paint_uniform_color(r_c)
    return mesh

def vis_mesh_o3d(vertices, faces):

    mesh = create_mesh(vertices, faces)
    min_y = -mesh.get_min_bound()[1]
    mesh.translate([0, min_y, 0])
    o3d.visualization.draw_geometries([mesh])
