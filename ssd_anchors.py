"""
SSD anchors
"""

import math
import numpy as np


def ssd_size_bounds_to_values(size_bounds, n_feat_layers, img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.

    The absolute values are measured in pixels, based on the network default size (300 pixels)

    Return:
        List of list containing the absolute sizes at each scale. For each scale,
        the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # start with the following smallest sizes
    sizes = [(img_size * size_bounds[0] / 2, img_size * size_bounds[0])]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100., img_size * (ratio + step) / 100.))   # sizes: [22.5, 45, 99, 153, 206, 315]
    return sizes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Compute SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative width and height.

    Args:
        feat_shape: Feature shape, used for computing relative position grids;
        sizes: Absolute reference sizes;
        ratios: Ratios to use on these features;
        img_shape: Image shape, used for computing height and width relatively to the former;
        offset: Grid offset.

    Return:
        y, x, h, w: Relative x and y grids, and height and width.
    """
    # compute the position grid (centers x & y)
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # expand dim to support easy broadcasting
    y = np.expand_dims(y, -1)   # [size, size, 1]
    x = np.expand_dims(x, -1)   # [size, size, 1]

    # compute the relative height and width
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors,), dtype=dtype)  # [n_anchors]
    w = np.zeros((num_anchors,), dtype=dtype)  # [n_anchors]
    # ratio = 1.0
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    start = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        start += 1
    for i, r in enumerate(ratios):
        h[start + i] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[start + i] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchor_all_layers(img_shape,
                          layers_shapes,
                          anchor_sizes,
                          anchor_ratios,
                          anchor_steps,
                          offset=0.5,
                          dtype=np.float32):
    """Compute anchor boxes for all feature layers
    """
    layers_anchors = []
    for i, layer_shape in enumerate(layers_shapes):
        anchor_bboxes = ssd_anchor_one_layer(img_shape,
                                             layer_shape,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset,
                                             dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors
