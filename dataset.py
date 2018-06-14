import tensorflow as tf
import Parameters

"""This function converts a Example protobuf object to a value in tensorflow """
def decode(serialized_example):
    
    features = tf.parse_single_example(
        serialized_example, 
        features={
            'expected_output': tf.FixedLenFeature([Parameters.NUMBER_OF_NETWORK_OUTPUTS], tf.float32), 
            'image': tf.FixedLenFeature([], tf.string)
        })

    print(features)

    image = tf.decode_raw(features['image'], tf.float32)    
    image = tf.reshape(image, [Parameters.IMAGE_SIZE, Parameters.IMAGE_SIZE, 3])

    label = tf.cast(features['expected_output'], tf.float32)
    #label = tf.one_hot(label, 2, 1.0, 0.0) #Convert from integer labels to one hot representation

    return image, label

def augment(image, label):
    #Any augmentation steps could be placed here
    return image, label

def normalize(image, label):
    #Convert to pseudo centered floats
    image = tf.cast(image, tf.float32) * (1.0/255) - .5

    return image, label

"""
    This function reads the given data source num_epoch times.
    @param batch_size: Number of examples to return
    @param num_epochs: Number of times to read the input data (0/None for forever)
    @param file_path: The file containing the examples

    
    @return iterator: .get_next() will return once it been intialized with sess.run(iterator.initializer) 
                      images: A float tensor with shape [batch_size, 128, 128, 3] in range [-0.5, 0.5]
                      A int32 tensor with shape [batch_size] with the true label in range [0, number_of_classes
"""
def GetInputs(batch_size, num_epochs, file_path):
    
    if not num_epochs:
        num_epochs = None
    
    with tf.name_scope('input'):
        #TFRecordDataset opens a protobuf file and reads entries line by line
        #Could also be a list of file file names
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode, num_parallel_calls=12)
        #dataset = dataset.map(augment, num_parallel_calls=12)
        dataset = dataset.map(normalize, num_parallel_calls=12)

        #The parameter is the queue size
        #dataset = dataset.shuffle(50 + 3*batch_size) #Not needed do to single epoc
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size*2)

        iterator = dataset.make_initializable_iterator()
    return iterator


