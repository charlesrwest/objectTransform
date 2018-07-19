import tensorflow as tf
import Parameters


session = tf.Session()
saver = tf.train.import_meta_graph('object_transform-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

op_to_restore = graph.get_tensor_by_name("res_fc_/dense/BiasAdd:0")


#Load images without convering to TFrecords

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.resize_images(image_decoded, [224, 224])
  return image_resized, label

# Get a vector of image paths


filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

