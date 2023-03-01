import time
import threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def main():
    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window('img', width=640, height=480)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget)    

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.5)
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()

    mat = rendering.Material()
    mat.shader = 'defaultLit'

    widget.scene.camera.look_at([0,0,0], [1,1,1], [0,0,1])
    widget.scene.add_geometry('frame', frame, mat)
    widget.scene.add_geometry('mesh', mesh, mat)

    def update_geometry():
        widget.scene.clear_geometry()
        widget.scene.add_geometry('frame', frame, mat)
        widget.scene.add_geometry('mesh', mesh, mat)   

    def thread_main():
        i = np.tile(np.arange(len(mesh.vertices)),(3,1)).T # (8,3)
        while True:
            # Deform mesh vertices
            vert = mesh.vertices + np.sin(i)*0.02
            mesh.vertices = o3d.utility.Vector3dVector(vert)
            i += 1

            # Update geometry
            gui.Application.instance.post_to_main_thread(window, update_geometry)            

            time.sleep(0.05)
            if i[0,0]>100:
                break

    threading.Thread(target=thread_main).start()

    gui.Application.instance.run()

if __name__ == "__main__":
    main()