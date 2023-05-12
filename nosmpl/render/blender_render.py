try:
    import bpy
    import os
    from math import radians
    from .cameras.blender_camera import PerspectiveCamera
    import numpy as np
    import logging
    from .blender_utils import stdout_redirected
    from nosmpl.utils.progress import prange
    import cv2
    from alfred.utils.file_io import get_new_video_writter
    # import gpu
    # import bgl

    # Set the logging level to ERROR or higher to disable logging output during rendering
    logging.basicConfig(level=logging.ERROR)
except ImportError as e:
    print(e)
    print("bpy not installed, will not work in this module.")
    pass


class BlenderRender:
    """
    BlenderRender using bpy as rendering
    you can choose CPU or GPU for render.
    GPU could takes more time but got higher quality.
    Which essentially same as in Blender App
    """

    def __init__(
        self,
        model_f,
        output_path=os.path.expanduser("~/tmp"),
        original_video_f=None,
        img_w=1024,
        img_h=960,
        # render_engine="cycles",
        render_engine="blender_eevee",
        device="gpu",
        to_video=True,
    ) -> None:
        preset_render_engines = ["blender_eevee", "blender_workbench", "cycles"]
        assert (
            render_engine in preset_render_engines
        ), f"render_engine must be one of: {preset_render_engines}"
        print(f"loading model to render: {model_f}")

        self.original_video_f = original_video_f
        if self.original_video_f is not None:
            self.cap = cv2.VideoCapture(self.original_video_f)
            # override given
            img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        else:
            self.cap = None

        self.resolution = (img_w, img_h)
        # create a scene
        self.scene = bpy.data.scenes.new("Scene")
        if "Cube" in bpy.data.objects:
            bpy.data.objects["Cube"].select_set(True)
            bpy.ops.object.delete()

        camera_data = bpy.data.cameras.new("Camera")
        camera_data.lens_unit = "FOV"
        camera_data.angle = radians(75)
        camera = bpy.data.objects.new("Camera", camera_data)
        # camera.location = (-2.0, 1.0, 1.0)
        camera.location = (0, -3, 1)
        camera.rotation_euler = [radians(a) for a in (90, 0.0, 0)]

        # bpy.ops.object.camera_add()
        # camera = bpy.data.objects["Camera"]
        # camera.name = 'Camera111'
        # # camera.data.sensor_fit = "HORIZONTAL"
        # # camera.data.sensor_width = img_w
        # # camera.data.sensor_height = img_h
        # camera.location = (-2.0, 1.0, 1.0)
        # camera.rotation_euler = [radians(a) for a in (0, 0.0, 0)]

        # camera.lens_unit = 'FOV'
        # camera.data.angle = 0.872665
        self.scene.collection.objects.link(camera)
        self.scene.camera = camera

        # camera = PerspectiveCamera(
        #     resolution=[1024, 960],
        #     fov_y=np.deg2rad(50),
        #     translation=[-2, 1, 1],
        #     # quaternion=[],
        # )
        # self.scene.camera = camera.blender_camera

        # import model
        bpy.ops.import_scene.fbx(filepath=model_f)
        bpy.context.scene.camera = camera

        self.scene.render.image_settings.file_format = "PNG"
        self.output_path = bpy.path.abspath(output_path)
        self.scene.render.filepath = output_path

        self.scene = bpy.data.scenes[0]

        # Render settings
        bpy.data.scenes[0].render.engine = render_engine.upper()
        bpy.context.scene.render.resolution_x = img_w
        bpy.context.scene.render.resolution_y = img_h
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.world.color = (0, 0, 0)
        if device == "gpu" and render_engine == "cycles":
            bpy.context.preferences.addons[
                render_engine
            ].preferences.compute_device_type = "CUDA"
            # Set the device and feature set
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.samples = 16
            # bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.use_adaptive_sampling = True
            # black background, for combine with origin image
        
        print(f"{self.scene.frame_start} {self.scene.frame_end}")

        fbx_obj = None
        for obj in bpy.data.objects:
            print(obj)
            if obj.animation_data:
                fbx_obj = obj
                break
        anim_data = fbx_obj.animation_data
        action = anim_data.action
        self.frame_start, self.frame_end = action.frame_range
        self.frame_start = int(self.frame_start)
        self.frame_end = int(self.frame_end)
        self.has_animation = self.frame_end - self.frame_start > 1
        print(f"{self.frame_start} {self.frame_end}")
        print(f"saved to: {output_path}, has_anim: {self.has_animation}")
        self.to_video = to_video
        


    def render_one_frame(self):
        # Render the scene
        with stdout_redirected():
            bpy.ops.render.render()
        out_p = self.output_path
        if os.path.isdir(out_p):
            out_p = os.path.join(self.output_path, "render_res.png")
        bpy.data.images["Render Result"].save_render(filepath=out_p)

    def render_all_frames(self):
        os.makedirs(self.output_path, exist_ok=True)
        print(f"render frames range: {self.frame_start}, {self.frame_end + 1}")
        if self.to_video:
            out_p = os.path.join(self.output_path, "render_res.mp4")
            vw = get_new_video_writter(
                new_width=self.resolution[0],
                new_height=self.resolution[1],
                fps=30,
                save_f=out_p,
            )
            
            for frame in prange(self.frame_start, self.frame_end + 1):
                # Set the frame to render
                self.scene.frame_set(frame)
                with stdout_redirected():
                    bpy.ops.render.render()

                out_p = os.path.join(self.output_path, f"{frame:04d}.png")
                bpy.data.images["Render Result"].save_render(filepath=out_p)
                pixels = cv2.imread(out_p, cv2.IMREAD_UNCHANGED)
                if self.cap is not None:
                    res, raw_img = self.cap.read()
                    if res:
                        fg_b, fg_g, fg_r, fg_a = cv2.split(pixels)
                        fg_a = fg_a / 255.0
                        label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a]).astype(np.uint8)
                        alpha = 0.5
                        blended_img = cv2.addWeighted(label_rgb, 0.8, raw_img, 0.8, 0.7)

                        # bg_b, bg_g, bg_r = cv2.split(raw_img)
                        # # Merge them back with opposite of the alpha channel
                        # raw_img_masked = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])
                        # blended_img = cv2.add(raw_img_masked, label_rgb)

                        vw.write(blended_img)
                else:
                    vw.write(pixels)
            vw.release()
            print("done")
        else:
            for frame in prange(self.frame_start, self.frame_end + 1):
                # Set the frame to render
                self.scene.frame_set(frame)
                with stdout_redirected():
                    bpy.ops.render.render()
                out_p = os.path.join(self.output_path, f"{frame:04d}.png")
                bpy.data.images["Render Result"].save_render(filepath=out_p)

    def render(self):
        if self.has_animation:
            self.render_all_frames()
        else:
            self.render_one_frame()

    def read_render_buffer_out(self, render_res):
        pass
        # gpu_tex = gpu.texture.from_image(render_result)
        # # Read image from GPU
        # gpu_tex.read()
        # # OR read image into a NumPy array (might be more convenient for later operations)
        # fbo = gpu.types.GPUFrameBuffer(color_slots=(gpu_tex,))

        # buffer_np = np.empty(gpu_tex.width * gpu_tex.height * 4, dtype=np.float32)
        # buffer = bgl.Buffer(bgl.GL_FLOAT, buffer_np.shape, buffer_np)
        # with fbo.bind():
        #     bgl.glReadBuffer(bgl.GL_BACK)
        #     bgl.glReadPixels(0, 0, gpu_tex.width, gpu_tex.height, bgl.GL_RGBA, bgl.GL_FLOAT, buffer)
