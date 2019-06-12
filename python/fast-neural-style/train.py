from __future__ import print_function
from __future__ import division
import tensorflow as tf
import os
import argparse
import time

from nets import nets_factory
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/wave.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    

    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not (os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("""Build Network""")
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False
            )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)