"""
Solving with box coordinates transformation between boxes
and original image.

This is essential but no-one make it clear. I am trying it.
"""
import numpy as np
import cv2


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop_bbox_info(img, center, scale, res=(224, 224)):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)
    if new_shape[0] < 1 or new_shape[1] < 1:
        return None, None, None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    if new_img.shape[0] == 0:
        return None, None, None

    # Compute bbox for Han's format
    bbox_scale_o2n = res[0] / new_img.shape[0]  # 224/ 531

    bboxTopLeft_inOriginal = (ul[0], ul[1])
    # viewer2D.ImShow(new_img.astype(np.uint8),name='original')
    return bbox_scale_o2n, np.array(bboxTopLeft_inOriginal)


def bbox_from_bbr(boxes_cxcywh, rescale=1.2, detection_thresh=0.2, imageHeight=None):
    """Get center and scale for bounding box from openpose detections."""
    # center = boxes_cxcywh[:2] + 0.5 * boxes_cxcywh[2:]
    center = boxes_cxcywh[:2]
    bbox_size = max(boxes_cxcywh[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale  # , bbox_XYWH


def get_box_scale_info(img, boxes_cxcywh, input_res=224):
    '''
    the boxes format is cxcyhw
    '''
    center, scale = bbox_from_bbr(boxes_cxcywh, imageHeight=img.shape[0])
    if center is None:
        return None, None, None, None, None

    box_scale_o2n, box_topleft = crop_bbox_info(
        img, center, scale, (input_res, input_res)
    )

    bboxInfo = {"center": center, "scale": scale, "boxes_cxcywh": boxes_cxcywh}
    return box_scale_o2n, box_topleft, bboxInfo


def convert_vertices_to_ori_img(
    data3D, scale, trans, box_scale_o2n, box_topleft, bAppTransFirst=False
):
    '''
    box_scale should calculated from how image cropped.
    and box_topleft is image cropped top left
    '''
    data3D = data3D.copy()
    resnet_input_size_half = 224 * 0.5
    if bAppTransFirst:  # Hand model
        data3D[:, 0:2] += trans
        data3D *= scale  # apply scaling
    else:
        data3D *= scale  # apply scaling
        data3D[:, 0:2] += trans

    # 112 is originated from hrm's input size (224,24)
    data3D *= resnet_input_size_half
    data3D /= box_scale_o2n

    if not isinstance(box_topleft, np.ndarray):
        assert isinstance(box_topleft, tuple)
        assert len(box_topleft) == 2
        box_topleft = np.array(box_topleft)
    data3D[:, :2] += box_topleft + resnet_input_size_half / box_scale_o2n
    return data3D
