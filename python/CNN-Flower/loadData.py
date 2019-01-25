import tensorflow as tf

import os
import urllib

url = "http://download.tensorflow.org/example_images/flower_photos.tgz"

filename = ''

def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(work_directory):
    tf.gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath