import bpy
import os
from math import radians
import sys
from nosmpl.render.blender_render import BlenderRender

# context = bpy.context


# model_p = os.path.abspath(sys.argv[1])
# print(f"loading model to render: {model_p}")
# # create a scene
# scene = bpy.data.scenes.new("Scene")
# camera_data = bpy.data.cameras.new("Camera")

# camera = bpy.data.objects.new("Camera", camera_data)
# camera.location = (-2.0, 1.0, 3.0) 
# camera.rotation_euler = [radians(a) for a in (0, 0.0, 0)]
# scene.collection.objects.link(camera)

# # do the same for lights etc
# # scene.update(
# scene.camera = camera

# # import model
# bpy.ops.import_scene.fbx(filepath=model_p)

# scene.render.image_settings.file_format = "PNG"
# output_path = bpy.path.abspath(os.path.join(os.path.dirname(__file__), "a.png"))
# scene.render.filepath = output_path

# bpy.data.scenes[0].render.engine = "CYCLES"
# bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

# # Set the device and feature set
# bpy.context.scene.cycles.device = "GPU"

# print(f"saved to: {output_path}")
# # Render the scene
# bpy.ops.render.render()
# bpy.data.images["Render Result"].save_render(filepath=output_path)

my_renderer = BlenderRender(
    sys.argv[1], output_path=os.path.join(os.path.dirname(__file__), "render_res"), original_video_f=sys.argv[2]
)
my_renderer.render()
