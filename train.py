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

#Prepare input data
num_channels = 3

def GetHeInitializer():
    return tf.contrib.layers.variance_scaling_initializer()

def AddResidualModule(inputVariable, outputDepth, stride, isTraining):
    conv1 = tf.layers.conv2d(inputs=inputVariable, filters=outputDepth, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer(), strides=(stride, stride))
    conv2 = tf.layers.conv2d(inputs=conv1, filters=outputDepth, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer())

    if (stride != 1) or (inputVariable.get_shape()[3] != outputDepth):
        skip_conv = tf.layers.conv2d(inputs=inputVariable, filters=outputDepth, kernel_size=[1, 1], padding="same", strides=(stride, stride), activation=tf.nn.relu, kernel_initializer=GetHeInitializer())
        sum_op = conv2 + skip_conv
        relu_sum = tf.nn.relu(sum_op)
        return relu_sum
    else:
        sum_op = conv2 + inputVariable
        relu_sum = tf.nn.relu(sum_op)
        return relu_sum


def ConstructNetwork(imageSize, numberOfChannels, numberOfOutputs, isTraining):
    input_placeholder = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, numberOfChannels], name='input')

    #Labels
    y_true = tf.placeholder(tf.float32, shape = [None, numberOfOutputs], name = 'y_true')

    layers = []

    with tf.variable_scope('conv0', reuse=tf.AUTO_REUSE):
        conv0 = tf.layers.conv2d(inputs=input_placeholder, filters=64, kernel_size=[7, 7], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer(), strides=(2,2))
        layers.append(conv0)

    with tf.variable_scope('maxpool0', reuse=tf.AUTO_REUSE):
        maxpool0 = tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[2,2], strides=2)
        layers.append(maxpool0)

    with tf.variable_scope("res0_", reuse=tf.AUTO_REUSE):
        network_head = AddResidualModule(layers[-1], 64, 1, isTraining)

    network_parameters = [[64, 1], [64, 1], [64, 1], [128, 2], [128, 1], [128, 1], [128, 1], [256, 2], [256, 1], [256, 1], [256, 1], [256, 1], [512, 2], [512, 1], [512, 1]]

    for index in range(0, len(network_parameters)):
            with tf.variable_scope("res" + str(index) + "_", reuse=tf.AUTO_REUSE):
                module_parameters = network_parameters[index]
                next_module = AddResidualModule(layers[-1], module_parameters[0], module_parameters[1], isTraining)
                layers.append(next_module)

    with tf.variable_scope("res_fc_", reuse=tf.AUTO_REUSE):
        flat = tf.reshape(layers[-1], [-1, layers[-1].get_shape()[1:4].num_elements()])
        fc0 = tf.layers.dense(inputs=flat, units=numberOfOutputs, kernel_initializer=GetHeInitializer())

        loss = tf.reduce_mean(tf.squared_difference(fc0, y_true))

        return input_placeholder, y_true, fc0, loss

def AddOptimizer(fc_network_output_op, loss_operator):

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=Parameters.INITIAL_TRAINING_RATE).minimize(loss_operator)
        return optimizer

def ComputeLoss(lossOp, dataSetIterator, inputVar, labelVar, session):
    epoch_count = 0
    loss_sum = 0.0
    session.run(dataSetIterator.initializer)
    images_op, labels_op = dataSetIterator.get_next()

    while True:
       try:
           images, labels = session.run([images_op, labels_op])

           loss_sum += session.run(lossOp, feed_dict={inputVar: images, labelVar: labels})
           epoch_count += 1
       except tf.errors.OutOfRangeError:
           break

    loss = loss_sum/epoch_count
    return loss

def ReportValidationLoss(lossOp, dataSetIterator, epoch, inputVar, labelVar, session):
    validation_loss = ComputeLoss(lossOp, dataSetIterator, inputVar, labelVar, session)

    message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Validation Loss: {1}"
    print(message.format(epoch, validation_loss))
    sys.stdout.flush()
    return validation_loss

def SaveOutputsAsJson(fileName, outputOp, dataSetIterator, inputVar, labelVar, session):
    session.run(dataSetIterator.initializer)
    images_op, labels_op = dataSetIterator.get_next()

    dictionary = {}

    batch_number = 0
    while True:
       try:
           images, labels = session.run([images_op, labels_op])

           output = session.run(outputOp, feed_dict={inputVar: images, labelVar: labels})

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


def TrainForNBatches(trainOp, lossOp, imagesOp, labelsOp, inputVar, labelVar, session, numberOfExamples):
    number_of_iterations = 0
    loss_sum = 0.0

    for example_index in range(0, numberOfExamples):    
#        try:
        images, labels = session.run([imagesOp, labelsOp])
#        except tf.errors.OutOfRangeError: # Let error happen, not suppose to hit data end here
            #session.run(dataSetIterator.initializer)
            #images_op, labels_op = dataSetIterator.get_next()
            

        [_, batch_loss] = session.run([trainOp, lossOp], feed_dict={inputVar: images, labelVar: labels})
        number_of_iterations += 1
        loss_sum += batch_loss
    loss = loss_sum/number_of_iterations

    return number_of_iterations, loss

def Train(numberOfEpochPerDataset, numberOfDatasets, checkpointPath, saver, trainingInput, trainingLabel, trainingOutput, trainingOp, trainingLoss, validationInput, validationLabel, validationOutput, validationLoss):
    training_log_file = open('trainingLog.csv', 'w')
    training_log_file.write('Epoc, Training Loss, Validation Loss\n')

    for dataset_index in range(0, numberOfDatasets):
        #Generate a new set of training data
        if(Parameters.REGENERATE_TRAINING_DATA):
            RegenerateTrainingData.RegenerateTrainingData("objectTransformDatasetTrain.tfrecords")

        #Setup reading from tfrecords file
        training_data_iterator = dataset.GetInputs(Parameters.BATCH_SIZE, 1, "/home/charlesrwest/storage/Datasets/objectTransform/objectTransformDatasetTrain.tfrecords")
        validation_data_iterator = dataset.GetInputs(Parameters.BATCH_SIZE, 1, "/home/charlesrwest/storage/Datasets/objectTransform/objectTransformDatasetValidate.tfrecords")    

        session.run(training_data_iterator.initializer)
        images_op, labels_op = training_data_iterator.get_next()

        for epoch in range(0, numberOfEpochPerDataset):
            #Training
            [_, training_loss] = TrainForNBatches(trainingOp, trainingLoss, images_op, labels_op, trainingInput, trainingLabel, session, Parameters.MAX_BATCHES_BEFORE_REPORTING)
            message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Training Loss: {1}"
            print(message.format(epoch, training_loss))
            
            sys.stdout.flush()

            #Validation and reporting
            validation_loss = ReportValidationLoss(validationLoss, validation_data_iterator, epoch, validationInput, validationLabel, session)
            SaveOutputsAsJson("results/results"+ str(epoch) +".json", validationOutput, validation_data_iterator, validationInput, validationLabel, session)
            message = "{0}, {1}, {2}\n"
            training_log_file.write(message.format(epoch, training_loss, validation_loss))
            training_log_file.flush()

            #Checkpoint model
            saver.save(session, './object_transform-model')

    training_log_file.close()

session = tf.Session()


#Make the network
training_input, training_label, training_output, training_loss = ConstructNetwork(Parameters.IMAGE_SIZE, num_channels, Parameters.NUMBER_OF_NETWORK_OUTPUTS, True)

validation_input, validation_label, validation_output, validation_loss = ConstructNetwork(Parameters.IMAGE_SIZE, num_channels, Parameters.NUMBER_OF_NETWORK_OUTPUTS, False)

session.run(tf.global_variables_initializer())

#Add the optimizer
optimizer = AddOptimizer(training_output, training_loss)

session.run(tf.global_variables_initializer())

#Train the network with checkpointing and logging
saver = tf.train.Saver()



Train(Parameters.NUMBER_OF_REPORT_CYCLES, Parameters.NUMBER_OF_DATA_GENERATION_CYCLES, './object_transform-model', saver, training_input, training_label, training_output, optimizer, training_loss, validation_input, validation_label, validation_output, validation_loss)




 


