import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/wave.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)