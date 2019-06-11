import tensorflow as tf
import yaml

slim = tf.contrib.slim


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS