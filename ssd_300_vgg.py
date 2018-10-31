"""
SSD net (vgg based) 300x300
"""

import numpy as np
from pprint import pprint
import tensorflow as tf
from collections import namedtuple
from ssd_layers import conv2d, max_pool2d, l2norm, dropout, pad2d, ssd_multibox_layer
from ssd_anchors import ssd_size_bounds_to_values, ssd_anchor_all_layers


# SSD parameters
ssd_params = namedtuple('ssd_params', ['img_shape',             # the input image size: 300x300
                                       'num_classes',           # number of classes: 20+1
                                       'no_annotation_label',
                                       'feat_layers',           # list of names of layers for detection
                                       'feat_shapes',           # list of feature map shapes of layers for detection
                                       'anchor_size_bounds',    # the min and max bounds of anchor sizes
                                       'anchor_sizes',          # list of anchor sizes of layers for detection
                                       'anchor_ratios',         # list of rations used in layers for detection
                                       'anchor_steps',          # list of cell size (pixel size) of layer for detection
                                       'anchor_offset',         # the center point offset
                                       'normalizations',        # list of normalizations of layers for detection
                                       'prior_scaling'])

class SSD(object):
    """SSD-Net-300"""
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.threshold = 0.5    # class score threshold
        self.ssd_params = ssd_params(img_shape=(300, 300),
                                     num_classes=21,
                                     no_annotation_label=21,
                                     feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
                                     feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                     anchor_size_bounds=[0.15, 0.90],
                                     anchor_sizes=[(21., 45.),
                                                   (45., 99.),
                                                   (99., 153.),
                                                   (153., 207.),
                                                   (207., 261.),
                                                   (261., 315.)],
                                     anchor_ratios=[[2.0, 0.5],
                                                    [2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [2.0, 0.5],
                                                    [2.0, 0.5]],
                                     anchor_steps=[8, 16, 32, 64, 100, 300],
                                     anchor_offset=0.5,
                                     normalizations=[20, -1, -1, -1, -1, -1],
                                     prior_scaling=[0.1, 0.1, 0.2, 0.2]
                                     )
        predictions, logits, locations = self._built_net()
        print('predictions_shape: {}'.format(predictions[0].get_shape()))
        print('logits_shape:      {}'.format(logits[0].get_shape()))
        print('locations_shape:   {}'.format(locations[0].get_shape()))
        # self._update_feat_shapes_from_net(predictions)
        classes, scores, bboxes = self._bboxes_select(predictions, locations)
        self._classes = classes
        self._scores = scores
        self._bboxes = bboxes

    def _built_net(self):
        """Construct SSD Net"""
        self.end_points = {}    # record the detection layers outputs
        self._images = tf.placeholder(tf.float32, shape=[None, self.ssd_params.img_shape[0], self.ssd_params.img_shape[1], 3])

        with tf.variable_scope('ssd_300_vgg'):
            # original vgg layers
            # block 1
            net = conv2d(self._images, 64, 3, scope='conv1_1')
            net = conv2d(net, 64, 3, scope='conv1_2')
            self.end_points['block1'] = net
            net = max_pool2d(net, 2, scope='pool1')
            # block 2
            net = conv2d(net, 128, 3, scope='conv2_1')
            net = conv2d(net, 128, 3, scope='conv2_2')
            self.end_points['block2'] = net
            net = max_pool2d(net, 2, scope='pool2')
            # block 3
            net = conv2d(net, 256, 3, scope='conv3_1')
            net = conv2d(net, 256, 3, scope='conv3_2')
            net = conv2d(net, 256, 3, scope='conv3_3')
            self.end_points['block3'] = net
            net = max_pool2d(net, 2, scope='pool3')
            # block 4
            net = conv2d(net, 512, 3, scope='conv4_1')
            net = conv2d(net, 512, 3, scope='conv4_2')
            net = conv2d(net, 512, 3, scope='conv4_3')
            self.end_points['block4'] = net
            net = max_pool2d(net, 2, scope='pool4')
            # block 55
            net = conv2d(net, 512, 3, scope='conv5_1')
            net = conv2d(net, 512, 3, scope='conv5_2')
            net = conv2d(net, 512, 3, scope='conv5_3')
            self.end_points['block5'] = net
            net = max_pool2d(net, 3, stride=1, scope='pool5')

            # additional SSD layers
            # block 6: use dilate conv
            net = conv2d(net, 1024, 3, dilation_rate=6, scope='conv6')
            self.end_points['block6'] = net
            # net = dropout(net, is_training=self.is_training)
            # block 7
            net = conv2d(net, 1024, 1, scope='conv7')
            self.end_points['block7'] = net
            # block 8
            net = conv2d(net, 256, 1, scope='conv8_1x1')
            net = conv2d(pad2d(net, 1), 512, 3, stride=2, scope='conv8_3x3', padding='valid')
            self.end_points['block8'] = net
            # block 9
            net = conv2d(net, 128, 1, scope='conv9_1x1')
            net = conv2d(pad2d(net, 1), 256, 3, stride=2, scope='conv9_3x3', padding='valid')
            self.end_points['block9'] = net
            # block 10
            net = conv2d(net, 128, 1, scope='conv10_1x1')
            net = conv2d(net, 256, 3, scope='conv10_3x3', padding='valid')
            self.end_points['block10'] = net
            # block 11
            net = conv2d(net, 128, 1, scope='conv11_1x1')
            net = conv2d(net, 256, 3, scope='conv11_3x3', padding='valid')
            self.end_points['block11'] = net

            pprint(self.end_points)

            # class and location predictions
            predictions = []
            logits = []
            locations = []
            for i, layer in enumerate(self.ssd_params.feat_layers):
                cls, loc = ssd_multibox_layer(self.end_points[layer],
                                              self.ssd_params.num_classes,
                                              self.ssd_params.anchor_sizes[i],
                                              self.ssd_params.anchor_ratios[i],
                                              self.ssd_params.normalizations[i],
                                              scope=layer + '_box')
                predictions.append(tf.nn.softmax(cls))  # [num_feat_layers, batch_size, feat_shape0, feat_shape1, num_anchors, num_classes]
                logits.append(cls)
                locations.append(loc)
            return predictions, logits, locations

    def _update_feat_shapes_from_net(self, predictions):
        """Obtain the parameter of feature shapes from the prediction layers"""
        new_feat_shapes = []
        for l in predictions:
            new_feat_shapes.append(l.get_shape().as_list()[1:])
        self.ssd_params._replace(feat_shapes=new_feat_shapes)

    def anchors(self):
        """Get SSD anchors"""
        return ssd_anchor_all_layers(self.ssd_params.img_shape,
                                     self.ssd_params.feat_shapes,
                                     self.ssd_params.anchor_sizes,
                                     self.ssd_params.anchor_ratios,
                                     self.ssd_params.anchor_steps,
                                     self.ssd_params.anchor_offset,
                                     np.float32)

    def _bboxes_decode_layer(self, feat_locations, anchor_bboxes, prior_scaling):
        """Decode the feat location of one layer

        Args:
            feat_locations: 5D tensor, [batch_size, size, size, n_anchors, 4]
            anchor_bboxes: list of tensors(y, x, h, w), shape: [size, size, 1], [size, size, 1], [n_anchors], [n_anchors]
            prior_scaling: list of 4 floats
        """
        y_ref, x_ref, h_ref, w_ref = anchor_bboxes
        # print(h_ref)
        cx = feat_locations[:, :, :, :, 0] * w_ref * prior_scaling[0] + x_ref
        cy = feat_locations[:, :, :, :, 1] * h_ref * prior_scaling[1] + y_ref
        w = w_ref * tf.exp(feat_locations[:, :, :, :, 2] * prior_scaling[2])
        h = w_ref * tf.exp(feat_locations[:, :, :, :, 3] * prior_scaling[3])
        # compute boxes coordinates (ymin, xmin, ymax, xmax)
        bboxes = tf.stack([cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0], axis=-1)
        # shape: [batch_size, size, size, n_anchors, 4]
        return bboxes

    def _bboxes_select_layer(self, feat_predictions, feat_locations, anchor_bboxes, prior_scaling):
        """Select boxes from the feat layer, only for batch_size=1"""
        n_bboxes = np.product(feat_predictions.get_shape().as_list()[1:-1])     # feat_shape0 * feat_shape1 * num_anchors_per_feat
        # decode the location
        bboxes = self._bboxes_decode_layer(feat_locations, anchor_bboxes, prior_scaling)
        # print(bboxes.get_shape())
        bboxes = tf.reshape(bboxes, [n_bboxes, 4])
        # print(bboxes.get_shape())
        predictions = tf.reshape(feat_predictions, [n_bboxes, self.ssd_params.num_classes])
        # print(feat_predictions.get_shape())
        # print(predictions.get_shape())
        # remove the background predictions
        sub_predictions = predictions[:, 1:]
        # print(sub_predictions.get_shape())
        # choose the max score class
        classes = tf.argmax(sub_predictions, axis=1) + 1    # class labels
        scores = tf.reduce_max(sub_predictions, axis=1)     # max_class scores
        # boxes selection: use threshold
        filter_mask = scores > self.threshold
        classes = tf.boolean_mask(classes, filter_mask)
        scores = tf.boolean_mask(scores, filter_mask)
        bboxes = tf.boolean_mask(bboxes, filter_mask)
        return classes, scores, bboxes

    def _bboxes_select(self, predictions, locations):
        """Select all bboxes predictions, only for batch_size=1"""
        anchor_bboxes_list = self.anchors()
        y, x, h, w = anchor_bboxes_list[0]
        print('anchor_bboxes_y_shape: {}'.format(y.shape))
        print('anchor_bboxes_x_shape: {}'.format(x.shape))
        print('anchor_bboxes_h_shape: {}'.format(h.shape))
        print('anchor_bboxes_w_shape: {}'.format(w.shape))
        classes_list = []
        scores_list = []
        bboxes_list = []
        # select bboxes for each feat layer
        for i in range(len(predictions)):
            anchor_bboxes = list(map(tf.convert_to_tensor, anchor_bboxes_list[i]))
            classes, scores, bboxes = self._bboxes_select_layer(predictions[i], locations[i], anchor_bboxes, self.ssd_params.prior_scaling)
            classes_list.append(classes)
            scores_list.append(scores)
            bboxes_list.append(bboxes)
        # combine all feat layers
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)
        return classes, scores, bboxes

    def images(self):
        return self._images

    def detection(self):
        return self._classes, self._scores, self._bboxes


if __name__ == '__main__':
    ssd = SSD()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './ssd/ssd_checkpoints/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt')
