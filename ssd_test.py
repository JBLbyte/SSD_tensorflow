"""
SSD test
"""

import cv2
import numpy as np
import tensorflow as tf
from ssd_300_vgg import SSD
import utils
import visualization as vis


def main():
    img = cv2.imread('images/street.jpg')
    img_origin = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img_origin.dtype)
    img = utils.preprocess_image(img_origin)
    print(img.dtype)
    img_batch = np.expand_dims(img, axis=0)
    print('batch of images shape: {}'.format(img_batch.shape))

    ssd = SSD()
    classes, scores, bboxes = ssd.detection()
    images = ssd.images()

    ckpt_filename = './model/ssd_checkpoints/ssd_vgg_300_weights.ckpt'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_filename)

        res_classes, res_scores, res_bboxes = sess.run([classes, scores, bboxes], feed_dict={images: img_batch})
        res_classes, res_scores, res_bboxes = utils.process_bboxes(res_classes, res_scores, res_bboxes)
        print('<classes>: \n{}\n'.format(res_classes))
        print('<scores>: \n{}\n'.format(res_scores))
        print('<bboxes>: \n{}\n'.format(res_bboxes))
        vis.plt_bboxes(img_origin, res_classes, res_scores, res_bboxes)


if __name__ == '__main__':
    main()
