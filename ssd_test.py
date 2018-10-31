"""
SSD test
"""
import tensorflow as tf
from ssd_300_vgg import SSD


def main():
    ssd = SSD()

    ckpt_filename = './ssd_checkpoints/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_filename)


if __name__ == '__main__':
    main()
