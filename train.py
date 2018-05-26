import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
from time import gmtime, strftime
import sys
import json
import RegenerateTrainingData
import Parameters

#Adding seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

batch_size = 16

#Prepare input data
img_size = 224
num_channels = 3

def GetHeInitializer():
    return tf.contrib.layers.variance_scaling_initializer()

def AddResidualModule(inputVariable, outputDepth, stride):
    conv1 = tf.layers.conv2d(inputs=inputVariable, filters=outputDepth, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer(), strides=(stride, stride))
    batchnorm1 = tf.layers.batch_normalization(conv1)
    conv2 = tf.layers.conv2d(inputs=batchnorm1, filters=outputDepth, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer())
    batchnorm2 = tf.layers.batch_normalization(conv2)

    if (stride != 1) or (inputVariable.get_shape()[3] != outputDepth):
        skip_conv = tf.layers.conv2d(inputs=inputVariable, filters=outputDepth, kernel_size=[1, 1], padding="same", strides=(stride, stride), activation=tf.nn.relu, kernel_initializer=GetHeInitializer())
        sum_op = batchnorm2 + skip_conv
        relu_sum = tf.nn.relu(sum_op)
        return relu_sum
    else:
        sum_op = batchnorm2 + inputVariable
        relu_sum = tf.nn.relu(sum_op)
        return relu_sum


def ConstructNetwork(imageSize, numberOfChannels, numberOfOutputs):
    input_placeholder = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, numberOfChannels], name='input')

    #Labels
    y_true = tf.placeholder(tf.float32, shape = [None, numberOfOutputs], name = 'y_true')

    layers = []

    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(inputs=input_placeholder, filters=64, kernel_size=[7, 7], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer(), strides=(2,2))
        layers.append(conv0)

    with tf.variable_scope('maxpool0'):
        maxpool0 = tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[2,2], strides=2)
        layers.append(maxpool0)

    network_head = AddResidualModule(layers[-1], 64, 1)

    network_parameters = [[64, 1], [64, 1], [64, 1], [128, 2], [128, 1], [128, 1], [128, 1], [256, 2], [256, 1], [256, 1], [256, 1], [256, 1], [512, 2], [512, 1], [512, 1]]

    for module_parameters in network_parameters:
        next_module = AddResidualModule(layers[-1], module_parameters[0], module_parameters[1])
        layers.append(next_module)

    flat = tf.reshape(layers[-1], [-1, layers[-1].get_shape()[1:4].num_elements()])
    fc0 = tf.layers.dense(inputs=flat, units=numberOfOutputs, kernel_initializer=GetHeInitializer())

    loss = tf.reduce_mean(tf.squared_difference(fc0, y_true))

    return input_placeholder, y_true, fc0, loss

def AddOptimizer(fc_network_output_op, loss_operator):

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_operator)
        return optimizer

def ComputeLoss(lossOp, dataSetIterator, session):
    epoch_count = 0
    loss_sum = 0.0
    session.run(dataSetIterator.initializer)
    images_op, labels_op = dataSetIterator.get_next()

    while True:
       try:
           images, labels = session.run([images_op, labels_op])

           loss_sum += session.run(lossOp, feed_dict={x: images, y_true: labels})
           epoch_count += 1
       except tf.errors.OutOfRangeError:
           break

    loss = loss_sum/epoch_count
    return loss

def ReportValidationLoss(lossOp, dataSetIterator, epoch, session):
    validation_loss = ComputeLoss(lossOp, dataSetIterator, session)

    message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Validation Loss: {1}"
    print(message.format(epoch, validation_loss))
    sys.stdout.flush()
    return validation_loss

def SaveOutputsAsJson(fileName, outputOp, dataSetIterator, session):
    session.run(dataSetIterator.initializer)
    images_op, labels_op = dataSetIterator.get_next()

    dictionary = {}

    batch_number = 0
    while True:
       try:
           images, labels = session.run([images_op, labels_op])

           output = session.run(outputOp, feed_dict={x: images, y_true: labels})

           for i in range(0, output.shape[0]-1):
               base_name = "example_" + str(batch_number) + "_" + str(i) + "_";
               dictionary[base_name+"label"] = labels[i, :].tolist()
               dictionary[base_name+"output"] = output[i, :].tolist()
           batch_number += 1
       except tf.errors.OutOfRangeError:
           break
    json_string = json.dumps(dictionary, sort_keys=True)

    json_file = open(fileName, "w")

    json_file.write(json_string)

    json_file.close()


def TrainForEpoch(trainOp, lossOp, dataSetIterator, session):
    number_of_iterations = 0
    loss_sum = 0.0
    session.run(dataSetIterator.initializer)
    images_op, labels_op = dataSetIterator.get_next()

    while True:    
        try:
            images, labels = session.run([images_op, labels_op])
        except tf.errors.OutOfRangeError:
            break

        [_, batch_loss] = session.run([trainOp, lossOp], feed_dict={x: images, y_true: labels})
        number_of_iterations += 1
        loss_sum += batch_loss
    loss = loss_sum/number_of_iterations

    return number_of_iterations, loss

def Train(numberOfEpochPerDataset, numberOfDatasets, checkpointPath, saver):
    training_log_file = open('trainingLog', 'w')
    training_log_file.write('Epoc, Training Loss, Validation Loss\n')

    for dataset_index in range(0, numberOfDatasets):
        #Generate a new set of training data
        RegenerateTrainingData.RegenerateTrainingData()

        #Setup reading from tfrecords file
        training_data_iterator = dataset.GetInputs(batch_size, 1, "/home/charlesrwest/cpp/Datasets/objectTransform/objectTransformDatasetTrain.tfrecords")
        validation_data_iterator = dataset.GetInputs(batch_size, 1, "/home/charlesrwest/cpp/Datasets/objectTransform/objectTransformDatasetValidate.tfrecords")    

        for epoch in range(0, numberOfEpochPerDataset):
            #Training
            [_, training_loss] = TrainForEpoch(optimizer, loss, training_data_iterator, session)
            message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Training Loss: {1}"
            print(message.format(epoch, training_loss))
            
            sys.stdout.flush()

            #Validation and reporting
            validation_loss = ReportValidationLoss(loss, validation_data_iterator, epoch, session)
            SaveOutputsAsJson("results/results"+ str(epoch) +".json", fc0, validation_data_iterator, session)
            message = "{0}, {1}, {2}\n"
            training_log_file.write(message.format(epoch, training_loss, validation_loss))
            training_log_file.flush()

            #Checkpoint model
            saver.save(session, './object_transform-model')

    training_log_file.close()

session = tf.Session()


#Make the network
x, y_true, fc0, loss = ConstructNetwork(img_size, num_channels, Parameters.NUMBER_OF_NETWORK_OUTPUTS)

session.run(tf.global_variables_initializer())

#Add the optimizer
optimizer = AddOptimizer(fc0, loss)

session.run(tf.global_variables_initializer())

#Train the network with checkpointing and logging
saver = tf.train.Saver()



Train(1, 1000000, './object_transform-model', saver)




 


