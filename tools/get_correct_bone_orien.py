import bpy

import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
from bpy_extras.io_utils import axis_conversion


def find_correction_matrix_recursive(bone, parent_correction_inv=None):
    parent = bone.parent
    # if parent and (parent.is_root):
    #     self.pre_matrix = settings.global_matrix

    if parent_correction_inv:
        pre_matrix = parent_correction_inv @ Matrix()
    else:
        pre_matrix = Matrix()

    correction_matrix = None

    # find best orientation to align bone with
    bone_children = tuple(child for child in bone.children)
    if len(bone_children) == 0:
        # no children, inherit the correction from parent (if possible)
        if parent is not None:
            correction_matrix = parent_correction_inv.inverted() if parent_correction_inv else None
    else:
        # else find how best to rotate the bone to align the Y axis with the children
        best_axis = (1, 0, 0)
        if len(bone_children) == 1:
            vec = (pre_matrix @ bone_children[0].matrix_local).to_translation()
            best_axis = Vector((0, 0, 1 if vec[2] >= 0 else -1))
            if abs(vec[0]) > abs(vec[1]):
                if abs(vec[0]) > abs(vec[2]):
                    best_axis = Vector((1 if vec[0] >= 0 else -1, 0, 0))
            elif abs(vec[1]) > abs(vec[2]):
                best_axis = Vector((0, 1 if vec[1] >= 0 else -1, 0))
        else:
            # get the child directions once because they may be checked several times
            child_locs = ((pre_matrix @ child.matrix_local).to_translation() for child in bone_children)
            child_locs = tuple(loc.normalized() for loc in child_locs if loc.magnitude > 0.0)

            # I'm not sure which one I like better...
            best_angle = -1.0
            for vec in child_locs:
                test_axis = Vector((0, 0, 1 if vec[2] >= 0 else -1))
                if abs(vec[0]) > abs(vec[1]):
                    if abs(vec[0]) > abs(vec[2]):
                        test_axis = Vector((1 if vec[0] >= 0 else -1, 0, 0))
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
        to_up = 'Z' if best_axis[2] >= 0 else '-Z'
        if abs(best_axis[0]) > abs(best_axis[1]):
            if abs(best_axis[0]) > abs(best_axis[2]):
                to_up = 'X' if best_axis[0] >= 0 else '-X'
        elif abs(best_axis[1]) > abs(best_axis[2]):
            to_up = 'Y' if best_axis[1] >= 0 else '-Y'
        to_forward = 'X' if to_up not in {'X', '-X'} else 'Y'

        # Build correction matrix
        if (to_up, to_forward) != ('Y', 'X'):
            correction_matrix = axis_conversion(from_forward='X',
                                                from_up='Y',
                                                to_forward=to_forward,
                                                to_up=to_up,
                                                ).to_4x4()

    print(f'{bone.name} correction_matrix: {correction_matrix}')
    # process children
    correction_matrix_inv = correction_matrix.inverted_safe() if correction_matrix else None
    for child in bone.children:
        # child is bone
        find_correction_matrix_recursive(child, correction_matrix_inv)


def get_bone_orientation(bone):
    '''
    get bone orient
    '''
    pass
    

bones = bpy.data.objects['Armature.003'].data.bones

print(len(bones))

find_correction_matrix_recursive(bones[0])

for b in bones:
#    print(b.matrix)
    print(b.matrix_local)
    children = b.children
    # print(f'{b.name}, has {len(children)} children')
    if children is not None:
        # print(children)
        pass
    
    parent = b.parent
    if parent is not None:
        # parent just only one
        print(f'{b.name}, has {parent} parent')
