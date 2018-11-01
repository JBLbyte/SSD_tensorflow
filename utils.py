"""
Help functions for SSD
"""


import cv2
import numpy as np


def show_image(img, window_name='image'):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def whiten_image(image, means=(123., 117., 104.)):
    """Subtracts the given means from each image channel"""
    if image.ndim != 3:
        raise ValueError('Input must be of shape: (height, width, channel>0).')
    num_channels = image.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must equal to the number of channels.')
    means = np.array(means, dtype=image.dtype)
    image = image - means
    return image


def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)


def preprocess_image(image):
    """Preprocess a image to inference"""
    image_cp = np.copy(image).astype(np.float32)
    # image_cp = whiten_image(image_cp)         # whiten the image
    image_resized = resize_image(image_cp)    # resize the image
    image_resized = np.array(image_resized, dtype=image.dtype)
    return image_resized


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox"""
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bboxes by decreasing order and keep only top_k"""
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_iou(bbox, bboxes):
    """Compute IOU between bbox1 and bbox2"""
    bbox = np.transpose(bbox)
    bboxes = np.transpose(bboxes)
    # bbox of intersection and volume
    ymin_inter = np.maximum(bbox[0], bboxes[0])
    xmin_inter = np.maximum(bbox[1], bboxes[1])
    ymax_inter = np.minimum(bbox[2], bboxes[2])
    xmax_inter = np.minimum(bbox[3], bboxes[3])
    h_inter = np.maximum(ymax_inter - ymin_inter, 0.0)
    w_inter = np.maximum(xmax_inter - xmin_inter, 0.0)
    volume_inter = h_inter * w_inter
    # volume of union
    vol1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    vol2 = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
    iou = volume_inter / (vol1 + vol2 - volume_inter)
    return iou


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes"""
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])     # compute overlap with bboxes
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_resize(bbox_ref, bboxes):
    """Resize bboxes based on a reference bbox. Assuming the bbox_ref is [0, 0, 1, 1]"""
    bboxes = np.copy(bboxes)
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes


def process_bboxes(classes, scores, bboxes, bbox_img=(0.0, 0.0, 1.0, 1.0), top_k=400, nms_threshold=0.5):
    """Process the bboxes including sort and nms"""
    bboxes = bboxes_clip(bbox_img, bboxes)
    classes, scores, bboxes = bboxes_sort(classes, scores, bboxes, top_k)
    classes, scores, bboxes = bboxes_nms(classes, scores, bboxes, nms_threshold)
    bboxes = bboxes_resize(bbox_img, bboxes)
    return classes, scores, bboxes
