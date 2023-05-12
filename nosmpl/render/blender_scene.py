try:
    import bpy
    from .cameras.blender_camera import Camera, PerspectiveCamera
except ImportError as e:
    pass


class Scene:
    def __init__(self):
        # Initialise Blender scene
        self._camera = None
        self._reset_scene()

    @staticmethod
    def _set_default_blender_parameters():
        # Setup scene parameters
        scene = bpy.data.scenes[0]
        scene.use_nodes = True
        bpy.context.scene.world.use_nodes = False
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.quality = 100
        bpy.context.scene.world.color = (0, 0, 0)
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.cycles.filter_width = 0  # turn off anti-aliasing
        # Important if you want to get a pure color background (eg. white background)
        bpy.context.scene.view_settings.view_transform = "Raw"
        bpy.context.scene.cycles.samples = (
            128  # Default value, can be changed in .render
        )

    @staticmethod
    def _remove_all_objects():
        """Removes all objects from the scene. Previously used to remove the default cube"""
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge()

    @staticmethod
    def _load_empty_scene():
        """Resets the scene to the empty state"""
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.outliner.orphans_purge()

    def _reset_scene(self):
        """Resets the scene to the empty state"""
        self._load_empty_scene()
        scene = bpy.data.scenes[0]
        scene.world = bpy.data.worlds.new("BlendifyWorld")
        self._set_default_blender_parameters()
        self._camera = None

    def clear(self):
        """Clears the scene"""
        self._reset_scene()

    @property
    def camera(self) -> Camera:
        return self._camera

    def set_perspective_camera(
        self,
        resolution,
        focal_dist: float = None,
        fov_x: float = None,
        fov_y: float = None,
        center=None,
        near: float = 0.1,
        far: float = 100.0,
        tag: str = "camera",
        quaternion=(1, 0, 0, 0),
        translation=(0, 0, 0),
        resolution_percentage: int = 100,
    ) -> PerspectiveCamera:
        """Set perspective camera in the scene. Replaces the previous scene camera, if it exists.
        One of focal_dist, fov_x or fov_y is required to set the camera parameters

        Args:
            resolution (Vector2di): (w, h), the resolution of the resulting image
            focal_dist (float, optional): Perspective Camera focal distance in millimeters (default: None)
            fov_x (float, optional): Camera lens horizontal field of view (default: None)
            fov_y (float, optional): Camera lens vertical field of view (default: None)
            center (Vector2d, optional): (x, y), horizontal and vertical shifts of the Camera (default: None)
            near (float, optional): Camera near clipping distance (default: 0.1)
            far (float, optional): Camera far clipping distance (default: 100)
            tag (str): name of the created object in Blender
            quaternion (Vector4d, optional): rotation applied to the Blender object (default: (1,0,0,0))
            translation (Vector3d, optional): translation applied to the Blender object (default: (0,0,0))
            resolution_percentage (int, optional):
        Returns:
            PerspectiveCamera: created camera
        """
        camera = PerspectiveCamera(
            resolution=resolution,
            focal_dist=focal_dist,
            fov_x=fov_x,
            fov_y=fov_y,
            center=center,
            near=near,
            far=far,
            tag=tag,
            quaternion=quaternion,
            translation=translation,
        )
        self._setup_camera(camera, resolution_percentage)
        return camera

    def _setup_camera(self, camera: Camera, resolution_percentage: int = 100):
        # Delete old camera
        if self._camera is not None:
            self._camera._blender_remove_object()
        # Set new camera
        self._camera = camera
        scene = bpy.data.scenes[0]
        scene.render.resolution_x = camera.resolution[0]
        scene.render.resolution_y = camera.resolution[1]
        scene.render.resolution_percentage = resolution_percentage
