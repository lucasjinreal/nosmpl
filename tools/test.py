### Import Utility class
class FbxImportHelperNode:
    """
    Temporary helper node to store a hierarchy of fbxNode objects before building Objects, Armatures and Bones.
    It tries to keep the correction data in one place so it can be applied consistently to the imported data.
    """

    __slots__ = (
        "_parent",
        "anim_compensation_matrix",
        "is_global_animation",
        "armature_setup",
        "armature",
        "bind_matrix",
        "bl_bone",
        "bl_data",
        "bl_obj",
        "bone_child_matrix",
        "children",
        "clusters",
        "fbx_elem",
        "fbx_name",
        "fbx_transform_data",
        "fbx_type",
        "is_armature",
        "has_bone_children",
        "is_bone",
        "is_root",
        "is_leaf",
        "matrix",
        "matrix_as_parent",
        "matrix_geom",
        "meshes",
        "post_matrix",
        "pre_matrix",
    )

    def __init__(self, fbx_elem, bl_data, fbx_transform_data, is_bone):
        self.fbx_name = (
            elem_name_ensure_class(fbx_elem, b"Model") if fbx_elem else "Unknown"
        )
        self.fbx_type = fbx_elem.props[2] if fbx_elem else None
        self.fbx_elem = fbx_elem
        self.bl_obj = None
        self.bl_data = bl_data
        self.bl_bone = None  # Name of bone if this is a bone (this may be different to fbx_name if there was a name conflict in Blender!)
        self.fbx_transform_data = fbx_transform_data
        self.is_root = False
        self.is_bone = is_bone
        self.is_armature = False
        self.armature = None  # For bones only, relevant armature node.
        self.has_bone_children = False  # True if the hierarchy below this node contains bones, important to support mixed hierarchies.
        self.is_leaf = False  # True for leaf-bones added to the end of some bone chains to set the lengths.
        self.pre_matrix = (
            None  # correction matrix that needs to be applied before the FBX transform
        )
        self.bind_matrix = None  # for bones this is the matrix used to bind to the skin
        if fbx_transform_data:
            (
                self.matrix,
                self.matrix_as_parent,
                self.matrix_geom,
            ) = blen_read_object_transform_do(fbx_transform_data)
        else:
            self.matrix, self.matrix_as_parent, self.matrix_geom = (None, None, None)
        self.post_matrix = (
            None  # correction matrix that needs to be applied after the FBX transform
        )
        self.bone_child_matrix = None  # Objects attached to a bone end not the beginning, this matrix corrects for that

        # XXX Those two are to handle the fact that rigged meshes are not linked to their armature in FBX, which implies
        #     that their animation is in global space (afaik...).
        #     This is actually not really solvable currently, since anim_compensation_matrix is not valid if armature
        #     itself is animated (we'd have to recompute global-to-local anim_compensation_matrix for each frame,
        #     and for each armature action... beyond being an insane work).
        #     Solution for now: do not read rigged meshes animations at all! sic...
        self.anim_compensation_matrix = None  # a mesh moved in the hierarchy may have a different local matrix. This compensates animations for this.
        self.is_global_animation = False

        self.meshes = None  # List of meshes influenced by this bone.
        self.clusters = []  # Deformer Cluster nodes
        self.armature_setup = {}  # mesh and armature matrix when the mesh was bound

        self._parent = None
        self.children = []

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            self._parent.children.remove(self)
        self._parent = value
        if self._parent is not None:
            self._parent.children.append(self)

    @property
    def ignore(self):
        # Separating leaf status from ignore status itself.
        # Currently they are equivalent, but this may change in future.
        return self.is_leaf

    def __repr__(self):
        if self.fbx_elem:
            return self.fbx_elem.props[1].decode()
        else:
            return "None"

    def print_info(self, indent=0):
        print(
            " " * indent
            + (self.fbx_name if self.fbx_name else "(Null)")
            + ("[root]" if self.is_root else "")
            + ("[leaf]" if self.is_leaf else "")
            + ("[ignore]" if self.ignore else "")
            + ("[armature]" if self.is_armature else "")
            + ("[bone]" if self.is_bone else "")
            + ("[HBC]" if self.has_bone_children else "")
        )
        for c in self.children:
            c.print_info(indent + 1)

    def mark_leaf_bones(self):
        if self.is_bone and len(self.children) == 1:
            child = self.children[0]
            if child.is_bone and len(child.children) == 0:
                child.is_leaf = True
        for child in self.children:
            child.mark_leaf_bones()

    def do_bake_transform(self, settings):
        return (
            settings.bake_space_transform
            and self.fbx_type in (b"Mesh", b"Null")
            and not self.is_armature
            and not self.is_bone
        )

    def find_correction_matrix(self, settings, parent_correction_inv=None):
        from bpy_extras.io_utils import axis_conversion

        if self.parent and (
            self.parent.is_root or self.parent.do_bake_transform(settings)
        ):
            self.pre_matrix = settings.global_matrix

        if parent_correction_inv:
            self.pre_matrix = parent_correction_inv @ (
                self.pre_matrix if self.pre_matrix else Matrix()
            )

        correction_matrix = None

        if self.is_bone:
            if settings.automatic_bone_orientation:
                # find best orientation to align bone with
                bone_children = tuple(child for child in self.children if child.is_bone)
                if len(bone_children) == 0:
                    # no children, inherit the correction from parent (if possible)
                    if self.parent and self.parent.is_bone:
                        correction_matrix = (
                            parent_correction_inv.inverted()
                            if parent_correction_inv
                            else None
                        )
                else:
                    # else find how best to rotate the bone to align the Y axis with the children
                    best_axis = (1, 0, 0)
                    if len(bone_children) == 1:
                        vec = bone_children[0].get_bind_matrix().to_translation()
                        best_axis = Vector((0, 0, 1 if vec[2] >= 0 else -1))
                        if abs(vec[0]) > abs(vec[1]):
                            if abs(vec[0]) > abs(vec[2]):
                                best_axis = Vector((1 if vec[0] >= 0 else -1, 0, 0))
                        elif abs(vec[1]) > abs(vec[2]):
                            best_axis = Vector((0, 1 if vec[1] >= 0 else -1, 0))
                    else:
                        # get the child directions once because they may be checked several times
                        child_locs = (
                            child.get_bind_matrix().to_translation()
                            for child in bone_children
                        )
                        child_locs = tuple(
                            loc.normalized()
                            for loc in child_locs
                            if loc.magnitude > 0.0
                        )

                        # I'm not sure which one I like better...
                        if False:
                            best_angle = -1.0
                            for i in range(6):
                                a = i // 2
                                s = -1 if i % 2 == 1 else 1
                                test_axis = Vector(
                                    (
                                        s if a == 0 else 0,
                                        s if a == 1 else 0,
                                        s if a == 2 else 0,
                                    )
                                )

                                # find max angle to children
                                max_angle = 1.0
                                for loc in child_locs:
                                    max_angle = min(max_angle, test_axis.dot(loc))

                                # is it better than the last one?
                                if best_angle < max_angle:
                                    best_angle = max_angle
                                    best_axis = test_axis
                        else:
                            best_angle = -1.0
                            for vec in child_locs:
                                test_axis = Vector((0, 0, 1 if vec[2] >= 0 else -1))
                                if abs(vec[0]) > abs(vec[1]):
                                    if abs(vec[0]) > abs(vec[2]):
                                        test_axis = Vector(
                                            (1 if vec[0] >= 0 else -1, 0, 0)
                                        )
                                elif abs(vec[1]) > abs(vec[2]):
                                    test_axis = Vector((0, 1 if vec[1] >= 0 else -1, 0))

                                # find max angle to children
                                max_angle = 1.0
                                for loc in child_locs:
                                    max_angle = min(max_angle, test_axis.dot(loc))

                                # is it better than the last one?
                                if best_angle < max_angle:
                                    best_angle = max_angle
                                    best_axis = test_axis

                    # convert best_axis to axis string
                    to_up = "Z" if best_axis[2] >= 0 else "-Z"
                    if abs(best_axis[0]) > abs(best_axis[1]):
                        if abs(best_axis[0]) > abs(best_axis[2]):
                            to_up = "X" if best_axis[0] >= 0 else "-X"
                    elif abs(best_axis[1]) > abs(best_axis[2]):
                        to_up = "Y" if best_axis[1] >= 0 else "-Y"
                    to_forward = "X" if to_up not in {"X", "-X"} else "Y"

                    # Build correction matrix
                    if (to_up, to_forward) != ("Y", "X"):
                        correction_matrix = axis_conversion(
                            from_forward="X",
                            from_up="Y",
                            to_forward=to_forward,
                            to_up=to_up,
                        ).to_4x4()
            else:
                correction_matrix = settings.bone_correction_matrix
        else:
            # camera and light can be hard wired
            if self.fbx_type == b"Camera":
                correction_matrix = MAT_CONVERT_CAMERA
            elif self.fbx_type == b"Light":
                correction_matrix = MAT_CONVERT_LIGHT

        self.post_matrix = correction_matrix

        if self.do_bake_transform(settings):
            self.post_matrix = settings.global_matrix_inv @ (
                self.post_matrix if self.post_matrix else Matrix()
            )

        # process children
        correction_matrix_inv = (
            correction_matrix.inverted_safe() if correction_matrix else None
        )
        for child in self.children:
            child.find_correction_matrix(settings, correction_matrix_inv)

    def find_armature_bones(self, armature):
        for child in self.children:
            if child.is_bone:
                child.armature = armature
                child.find_armature_bones(armature)

    def find_armatures(self):
        needs_armature = False
        for child in self.children:
            if child.is_bone:
                needs_armature = True
                break
        if needs_armature:
            if self.fbx_type in {b"Null", b"Root"}:
                # if empty then convert into armature
                self.is_armature = True
                armature = self
            else:
                # otherwise insert a new node
                # XXX Maybe in case self is virtual FBX root node, we should instead add one armature per bone child?
                armature = FbxImportHelperNode(None, None, None, False)
                armature.fbx_name = "Armature"
                armature.is_armature = True

                for child in tuple(self.children):
                    if child.is_bone:
                        child.parent = armature

                armature.parent = self

            armature.find_armature_bones(armature)

        for child in self.children:
            if child.is_armature or child.is_bone:
                continue
            child.find_armatures()

    def find_bone_children(self):
        has_bone_children = False
        for child in self.children:
            has_bone_children |= child.find_bone_children()
        self.has_bone_children = has_bone_children
        return self.is_bone or has_bone_children

    def find_fake_bones(self, in_armature=False):
        if in_armature and not self.is_bone and self.has_bone_children:
            self.is_bone = True
            # if we are not a null node we need an intermediate node for the data
            if self.fbx_type not in {b"Null", b"Root"}:
                node = FbxImportHelperNode(self.fbx_elem, self.bl_data, None, False)
                self.fbx_elem = None
                self.bl_data = None

                # transfer children
                for child in self.children:
                    if child.is_bone or child.has_bone_children:
                        continue
                    child.parent = node

                # attach to parent
                node.parent = self

        if self.is_armature:
            in_armature = True
        for child in self.children:
            child.find_fake_bones(in_armature)

    def get_world_matrix_as_parent(self):
        matrix = self.parent.get_world_matrix_as_parent() if self.parent else Matrix()
        if self.matrix_as_parent:
            matrix = matrix @ self.matrix_as_parent
        return matrix

    def get_world_matrix(self):
        matrix = self.parent.get_world_matrix_as_parent() if self.parent else Matrix()
        if self.matrix:
            matrix = matrix @ self.matrix
        return matrix

    def get_matrix(self):
        matrix = self.matrix if self.matrix else Matrix()
        if self.pre_matrix:
            matrix = self.pre_matrix @ matrix
        if self.post_matrix:
            matrix = matrix @ self.post_matrix
        return matrix

    def get_bind_matrix(self):
        matrix = self.bind_matrix if self.bind_matrix else Matrix()
        if self.pre_matrix:
            matrix = self.pre_matrix @ matrix
        if self.post_matrix:
            matrix = matrix @ self.post_matrix
        return matrix

    def make_bind_pose_local(self, parent_matrix=None):
        if parent_matrix is None:
            parent_matrix = Matrix()

        if self.bind_matrix:
            bind_matrix = parent_matrix.inverted_safe() @ self.bind_matrix
        else:
            bind_matrix = self.matrix.copy() if self.matrix else None

        self.bind_matrix = bind_matrix
        if bind_matrix:
            parent_matrix = parent_matrix @ bind_matrix

        for child in self.children:
            child.make_bind_pose_local(parent_matrix)

    def collect_skeleton_meshes(self, meshes):
        for _, m in self.clusters:
            meshes.update(m)
        for child in self.children:
            if not child.meshes:
                child.collect_skeleton_meshes(meshes)

    def collect_armature_meshes(self):
        if self.is_armature:
            armature_matrix_inv = self.get_world_matrix().inverted_safe()

            meshes = set()
            for child in self.children:
                # Children meshes may be linked to children armatures, in which case we do not want to link them
                # to a parent one. See T70244.
                child.collect_armature_meshes()
                if not child.meshes:
                    child.collect_skeleton_meshes(meshes)
            for m in meshes:
                old_matrix = m.matrix
                m.matrix = armature_matrix_inv @ m.get_world_matrix()
                m.anim_compensation_matrix = old_matrix.inverted_safe() @ m.matrix
                m.is_global_animation = True
                m.parent = self
            self.meshes = meshes
        else:
            for child in self.children:
                child.collect_armature_meshes()

    def build_skeleton(
        self, arm, parent_matrix, parent_bone_size=1, force_connect_children=False
    ):
        def child_connect(par_bone, child_bone, child_head, connect_ctx):
            # child_bone or child_head may be None.
            force_connect_children, connected = connect_ctx
            if child_bone is not None:
                child_bone.parent = par_bone
                child_head = child_bone.head

            if similar_values_iter(par_bone.tail, child_head):
                if child_bone is not None:
                    child_bone.use_connect = True
                # Disallow any force-connection at this level from now on, since that child was 'really'
                # connected, we do not want to move current bone's tail anymore!
                connected = None
            elif force_connect_children and connected is not None:
                # We only store position where tail of par_bone should be in the end.
                # Actual tail moving and force connection of compatible child bones will happen
                # once all have been checked.
                if connected is ...:
                    connected = (
                        [child_head.copy(), 1],
                        [child_bone] if child_bone is not None else [],
                    )
                else:
                    connected[0][0] += child_head
                    connected[0][1] += 1
                    if child_bone is not None:
                        connected[1].append(child_bone)
            connect_ctx[1] = connected

        def child_connect_finalize(par_bone, connect_ctx):
            force_connect_children, connected = connect_ctx
            # Do nothing if force connection is not enabled!
            if (
                force_connect_children
                and connected is not None
                and connected is not ...
            ):
                # Here again we have to be wary about zero-length bones!!!
                par_tail = connected[0][0] / connected[0][1]
                if (par_tail - par_bone.head).magnitude < 1e-2:
                    par_bone_vec = (par_bone.tail - par_bone.head).normalized()
                    par_tail = par_bone.head + par_bone_vec * 0.01
                par_bone.tail = par_tail
                for child_bone in connected[1]:
                    if similar_values_iter(par_tail, child_bone.head):
                        child_bone.use_connect = True

        # Create the (edit)bone.
        bone = arm.bl_data.edit_bones.new(name=self.fbx_name)
        bone.select = True
        self.bl_obj = arm.bl_obj
        self.bl_data = arm.bl_data
        self.bl_bone = bone.name  # Could be different from the FBX name!

        # get average distance to children
        bone_size = 0.0
        bone_count = 0
        for child in self.children:
            if child.is_bone:
                bone_size += child.get_bind_matrix().to_translation().magnitude
                bone_count += 1
        if bone_count > 0:
            bone_size /= bone_count
        else:
            bone_size = parent_bone_size

        # So that our bone gets its final length, but still Y-aligned in armature space.
        # 0-length bones are automatically collapsed into their parent when you leave edit mode,
        # so this enforces a minimum length.
        bone_tail = Vector((0.0, 1.0, 0.0)) * max(0.01, bone_size)
        bone.tail = bone_tail

        # And rotate/move it to its final "rest pose".
        bone_matrix = parent_matrix @ self.get_bind_matrix().normalized()

        bone.matrix = bone_matrix

        # Correction for children attached to a bone. FBX expects to attach to the head of a bone,
        # while Blender attaches to the tail.
        self.bone_child_matrix = Matrix.Translation(-bone_tail)

        connect_ctx = [force_connect_children, ...]
        for child in self.children:
            if child.is_leaf and force_connect_children:
                # Arggggggggggggggggg! We do not want to create this bone, but we need its 'virtual head' location
                # to orient current one!!!
                child_head = (
                    bone_matrix @ child.get_bind_matrix().normalized()
                ).translation
                child_connect(bone, None, child_head, connect_ctx)
            elif child.is_bone and not child.ignore:
                child_bone = child.build_skeleton(
                    arm,
                    bone_matrix,
                    bone_size,
                    force_connect_children=force_connect_children,
                )
                # Connection to parent.
                child_connect(bone, child_bone, None, connect_ctx)

        child_connect_finalize(bone, connect_ctx)
        return bone

    def build_node_obj(self, fbx_tmpl, settings):
        if self.bl_obj:
            return self.bl_obj

        if self.is_bone or not self.fbx_elem:
            return None

        # create when linking since we need object data
        elem_name_utf8 = self.fbx_name

        # Object data must be created already
        self.bl_obj = obj = bpy.data.objects.new(
            name=elem_name_utf8, object_data=self.bl_data
        )

        fbx_props = (
            elem_find_first(self.fbx_elem, b"Properties70"),
            elem_find_first(fbx_tmpl, b"Properties70", fbx_elem_nil),
        )

        # ----
        # Misc Attributes

        obj.color[0:3] = elem_props_get_color_rgb(fbx_props, b"Color", (0.8, 0.8, 0.8))
        obj.hide_viewport = not bool(
            elem_props_get_visibility(fbx_props, b"Visibility", 1.0)
        )

        obj.matrix_basis = self.get_matrix()

        if settings.use_custom_props:
            blen_read_custom_properties(self.fbx_elem, obj, settings)

        return obj

    def build_skeleton_children(self, fbx_tmpl, settings, scene, view_layer):
        if self.is_bone:
            for child in self.children:
                if child.ignore:
                    continue
                child.build_skeleton_children(fbx_tmpl, settings, scene, view_layer)
            return None
        else:
            # child is not a bone
            obj = self.build_node_obj(fbx_tmpl, settings)

            if obj is None:
                return None

            for child in self.children:
                if child.ignore:
                    continue
                child.build_skeleton_children(fbx_tmpl, settings, scene, view_layer)

            # instance in scene
            view_layer.active_layer_collection.collection.objects.link(obj)
            obj.select_set(True)

            return obj

    def link_skeleton_children(self, fbx_tmpl, settings, scene):
        if self.is_bone:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.bl_obj
                if child_obj and child_obj != self.bl_obj:
                    child_obj.parent = (
                        self.bl_obj
                    )  # get the armature the bone belongs to
                    child_obj.parent_bone = self.bl_bone
                    child_obj.parent_type = "BONE"
                    child_obj.matrix_parent_inverse = Matrix()

                    # Blender attaches to the end of a bone, while FBX attaches to the start.
                    # bone_child_matrix corrects for that.
                    if child.pre_matrix:
                        child.pre_matrix = self.bone_child_matrix @ child.pre_matrix
                    else:
                        child.pre_matrix = self.bone_child_matrix

                    child_obj.matrix_basis = child.get_matrix()
                child.link_skeleton_children(fbx_tmpl, settings, scene)
            return None
        else:
            obj = self.bl_obj

            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.link_skeleton_children(fbx_tmpl, settings, scene)
                if child_obj:
                    child_obj.parent = obj

            return obj

    def set_pose_matrix(self, arm):
        pose_bone = arm.bl_obj.pose.bones[self.bl_bone]
        pose_bone.matrix_basis = (
            self.get_bind_matrix().inverted_safe() @ self.get_matrix()
        )

        for child in self.children:
            if child.ignore:
                continue
            if child.is_bone:
                child.set_pose_matrix(arm)

    def merge_weights(self, combined_weights, fbx_cluster):
        indices = elem_prop_first(
            elem_find_first(fbx_cluster, b"Indexes", default=None), default=()
        )
        weights = elem_prop_first(
            elem_find_first(fbx_cluster, b"Weights", default=None), default=()
        )

        for index, weight in zip(indices, weights):
            w = combined_weights.get(index)
            if w is None:
                combined_weights[index] = [weight]
            else:
                w.append(weight)

    def set_bone_weights(self):
        ignored_children = tuple(
            child
            for child in self.children
            if child.is_bone and child.ignore and len(child.clusters) > 0
        )

        if len(ignored_children) > 0:
            # If we have an ignored child bone we need to merge their weights into the current bone weights.
            # This can happen both intentionally and accidentally when skinning a model. Either way, they
            # need to be moved into a parent bone or they cause animation glitches.
            for fbx_cluster, meshes in self.clusters:
                combined_weights = {}
                self.merge_weights(combined_weights, fbx_cluster)

                for child in ignored_children:
                    for child_cluster, child_meshes in child.clusters:
                        if not meshes.isdisjoint(child_meshes):
                            self.merge_weights(combined_weights, child_cluster)

                # combine child weights
                indices = []
                weights = []
                for i, w in combined_weights.items():
                    indices.append(i)
                    if len(w) > 1:
                        weights.append(sum(w) / len(w))
                    else:
                        weights.append(w[0])

                add_vgroup_to_objects(
                    indices, weights, self.bl_bone, [node.bl_obj for node in meshes]
                )

            # clusters that drive meshes not included in a parent don't need to be merged
            all_meshes = set().union(*[meshes for _, meshes in self.clusters])
            for child in ignored_children:
                for child_cluster, child_meshes in child.clusters:
                    if all_meshes.isdisjoint(child_meshes):
                        indices = elem_prop_first(
                            elem_find_first(child_cluster, b"Indexes", default=None),
                            default=(),
                        )
                        weights = elem_prop_first(
                            elem_find_first(child_cluster, b"Weights", default=None),
                            default=(),
                        )
                        add_vgroup_to_objects(
                            indices,
                            weights,
                            self.bl_bone,
                            [node.bl_obj for node in child_meshes],
                        )
        else:
            # set the vertex weights on meshes
            for fbx_cluster, meshes in self.clusters:
                indices = elem_prop_first(
                    elem_find_first(fbx_cluster, b"Indexes", default=None), default=()
                )
                weights = elem_prop_first(
                    elem_find_first(fbx_cluster, b"Weights", default=None), default=()
                )
                add_vgroup_to_objects(
                    indices, weights, self.bl_bone, [node.bl_obj for node in meshes]
                )

        for child in self.children:
            if child.is_bone and not child.ignore:
                child.set_bone_weights()

    def build_hierarchy(self, fbx_tmpl, settings, scene, view_layer):
        if self.is_armature:
            # create when linking since we need object data
            elem_name_utf8 = self.fbx_name

            self.bl_data = arm_data = bpy.data.armatures.new(name=elem_name_utf8)

            # Object data must be created already
            self.bl_obj = arm = bpy.data.objects.new(
                name=elem_name_utf8, object_data=arm_data
            )

            arm.matrix_basis = self.get_matrix()

            if self.fbx_elem:
                fbx_props = (
                    elem_find_first(self.fbx_elem, b"Properties70"),
                    elem_find_first(fbx_tmpl, b"Properties70", fbx_elem_nil),
                )

                if settings.use_custom_props:
                    blen_read_custom_properties(self.fbx_elem, arm, settings)

            # instance in scene
            view_layer.active_layer_collection.collection.objects.link(arm)
            arm.select_set(True)

            # Add bones:

            # Switch to Edit mode.
            view_layer.objects.active = arm
            is_hidden = arm.hide_viewport
            arm.hide_viewport = False  # Can't switch to Edit mode hidden objects...
            bpy.ops.object.mode_set(mode="EDIT")

            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.build_skeleton(
                        self,
                        Matrix(),
                        force_connect_children=settings.force_connect_children,
                    )

            bpy.ops.object.mode_set(mode="OBJECT")

            arm.hide_viewport = is_hidden

            # Set pose matrix
            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.set_pose_matrix(self)

            # Add bone children:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.build_skeleton_children(
                    fbx_tmpl, settings, scene, view_layer
                )

            return arm
        elif self.fbx_elem and not self.is_bone:
            obj = self.build_node_obj(fbx_tmpl, settings)

            # walk through children
            for child in self.children:
                child.build_hierarchy(fbx_tmpl, settings, scene, view_layer)

            # instance in scene
            view_layer.active_layer_collection.collection.objects.link(obj)
            obj.select_set(True)

            return obj
        else:
            for child in self.children:
                child.build_hierarchy(fbx_tmpl, settings, scene, view_layer)

            return None

    def link_hierarchy(self, fbx_tmpl, settings, scene):
        if self.is_armature:
            arm = self.bl_obj

            # Link bone children:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.link_skeleton_children(fbx_tmpl, settings, scene)
                if child_obj:
                    child_obj.parent = arm

            # Add armature modifiers to the meshes
            if self.meshes:
                for mesh in self.meshes:
                    (mmat, amat) = mesh.armature_setup[self]
                    me_obj = mesh.bl_obj

                    # bring global armature & mesh matrices into *Blender* global space.
                    # Note: Usage of matrix_geom (local 'diff' transform) here is quite brittle.
                    #       Among other things, why in hell isn't it taken into account by bindpose & co???
                    #       Probably because org app (max) handles it completely aside from any parenting stuff,
                    #       which we obviously cannot do in Blender. :/
                    if amat is None:
                        amat = self.bind_matrix
                    amat = settings.global_matrix @ (Matrix() if amat is None else amat)
                    if self.matrix_geom:
                        amat = amat @ self.matrix_geom
                    mmat = settings.global_matrix @ mmat
                    if mesh.matrix_geom:
                        mmat = mmat @ mesh.matrix_geom

                    # Now that we have armature and mesh in there (global) bind 'state' (matrix),
                    # we can compute inverse parenting matrix of the mesh.
                    me_obj.matrix_parent_inverse = (
                        amat.inverted_safe()
                        @ mmat
                        @ me_obj.matrix_basis.inverted_safe()
                    )

                    mod = mesh.bl_obj.modifiers.new(arm.name, "ARMATURE")
                    mod.object = arm

            # Add bone weights to the deformers
            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.set_bone_weights()

            return arm
        elif self.bl_obj:
            obj = self.bl_obj

            # walk through children
            for child in self.children:
                child_obj = child.link_hierarchy(fbx_tmpl, settings, scene)
                if child_obj:
                    child_obj.parent = obj

            return obj
        else:
            for child in self.children:
                child.link_hierarchy(fbx_tmpl, settings, scene)

            return None
