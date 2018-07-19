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


def ConstructNetwork(inputOp, labelOp, imageSize, numberOfChannels, numberOfOutputs):

    is_training_placeholder = tf.placeholder(tf.bool, shape=[1], name='is_training')

    layers = []

    with tf.variable_scope('conv0', reuse=tf.AUTO_REUSE):
        conv0 = tf.layers.conv2d(inputs=inputOp, filters=64, kernel_size=[7, 7], padding="same", activation=tf.nn.relu, kernel_initializer=GetHeInitializer(), strides=(2,2))
        layers.append(conv0)

    with tf.variable_scope('maxpool0', reuse=tf.AUTO_REUSE):
        maxpool0 = tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[2,2], strides=2)
        layers.append(maxpool0)

    with tf.variable_scope("res0_", reuse=tf.AUTO_REUSE):
        network_head = AddResidualModule(layers[-1], 64, 1, is_training_placeholder)

    network_parameters = [[64, 1], [64, 1], [64, 1], [128, 2], [128, 1], [128, 1], [128, 1], [256, 2], [256, 1], [256, 1], [256, 1], [256, 1], [512, 2], [512, 1], [512, 1]]

    for index in range(0, len(network_parameters)):
            with tf.variable_scope("res" + str(index) + "_", reuse=tf.AUTO_REUSE):
                module_parameters = network_parameters[index]
                next_module = AddResidualModule(layers[-1], module_parameters[0], module_parameters[1], is_training_placeholder)
                layers.append(next_module)

    with tf.variable_scope("res_fc_", reuse=tf.AUTO_REUSE):
        flat = tf.reshape(layers[-1], [-1, layers[-1].get_shape()[1:4].num_elements()])
        fc0 = tf.layers.dense(inputs=flat, units=numberOfOutputs, kernel_initializer=GetHeInitializer())

        loss = tf.reduce_mean(tf.squared_difference(fc0, labelOp))

        print("Output name: " + str(fc0))

        return is_training_placeholder, fc0, loss

def AddOptimizer(fc_network_output_op, loss_operator):

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=Parameters.INITIAL_TRAINING_RATE).minimize(loss_operator)
        return optimizer

def ComputeLoss(lossOp, dataSetInitializer, session):
    epoch_count = 0
    loss_sum = 0.0
    session.run(dataSetInitializer)

    while True:
       try:
           loss_sum += session.run(lossOp)
           epoch_count += 1
       except tf.errors.OutOfRangeError:
           break

    loss = loss_sum/epoch_count
    return loss

def ReportValidationLoss(lossOp, dataSetInitializer, epoch, session):
    validation_loss = ComputeLoss(lossOp, dataSetInitializer, session)

    message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Validation Loss: {1}"
    print(message.format(epoch, validation_loss))
    sys.stdout.flush()
    return validation_loss

def SaveOutputsAsJson(fileName, outputOp, lossOp, dataSetInitializer, labelOp, imageNameOp, session):
    session.run(dataSetInitializer)

    dictionary = {}

    batch_number = 0
    while True:
       try:
           output, labels, image_names, losses = session.run([outputOp, labelOp, imageNameOp, lossOp])

           for i in range(0, output.shape[0]):
               base_name = image_names[i, 0].decode('UTF-8')
               dictionary[base_name+"_label"] = labels[i, :].tolist()
               dictionary[base_name+"_output"] = output[i, :].tolist()
               dictionary[base_name+"_loss"] = losses.tolist()
           batch_number += 1
       except tf.errors.OutOfRangeError:
           break
    json_string = json.dumps(dictionary, sort_keys=True)

    json_file = open(fileName, "w")

    json_file.write(json_string)

    json_file.close()


def TrainForNBatches(trainOp, lossOp, datasetInitializer, session, numberOfBatches):
    number_of_iterations = 0
    loss_sum = 0.0

    for example_index in range(0, numberOfBatches):    
        try:
            [_, batch_loss] = session.run([trainOp, lossOp])
            number_of_iterations += 1
            loss_sum += batch_loss
        except tf.errors.OutOfRangeError: # Let error happen, not suppose to hit data end here
            session.run(datasetInitializer)
            example_index = example_index -1

    loss = loss_sum/number_of_iterations

    return number_of_iterations, loss

def Train(numberOfEpochPerDataset, numberOfDatasets, checkpointPath, saver, outputOp, trainingOp, lossOp, dataIterator, validationDatasetInitOp, isTrainingPlaceHolder, labelOp, imageNameOp):
    old_validation_loss = sys.float_info.max;
    training_log_file = open('trainingLog.csv', 'w')
    training_log_file.write('Epoc, Training Loss, Validation Loss\n')

    for dataset_index in range(0, numberOfDatasets):
        #Generate a new set of training data
        if(Parameters.REGENERATE_TRAINING_DATA):
            RegenerateTrainingData.RegenerateTrainingData("objectTransformDatasetTrain.tfrecords")

        #Setup reading from tfrecords file
        training_dataset = dataset.GetDataset(Parameters.BATCH_SIZE, 1, "/home/charlesrwest/storage/Datasets/objectTransform/objectTransformDatasetTrain.tfrecords") 

        # create the initialisation operations
        train_init_op = dataIterator.make_initializer(training_dataset)

        session.run(train_init_op)

        for epoch in range(0, numberOfEpochPerDataset):
            #Training
            [_, training_loss] = TrainForNBatches(trainingOp, lossOp, train_init_op, session, Parameters.MAX_BATCHES_BEFORE_REPORTING)
            message = "Training Epoch {0} --- " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" --- Training Loss: {1}"
            print(message.format(epoch, training_loss))
            
            sys.stdout.flush()

            #Validation and reporting
            validation_loss = ReportValidationLoss(lossOp, validationDatasetInitOp, epoch, session)
            SaveOutputsAsJson("results/results"+ str(epoch) +".json", outputOp, lossOp, validationDatasetInitOp, labelOp, imageNameOp, session)
            message = "{0}, {1}, {2}\n"
            training_log_file.write(message.format(epoch, training_loss, validation_loss))
            training_log_file.flush()

            #Checkpoint model if the validation loss is better
            if validation_loss < old_validation_loss:
                old_validation_loss = validation_loss
                saver.save(session, './object_transform-model')
                print("Validation error improved, so checkpointed")

    training_log_file.close()

session = tf.Session()

validation_dataset = dataset.GetDataset(1, 1, "/home/charlesrwest/storage/Datasets/objectTransform/objectTransformDatasetValidate.tfrecords")   

iterator = tf.data.Iterator.from_structure(validation_dataset.output_types,
                                           validation_dataset.output_shapes)
images, image_names, labels  = iterator.get_next()

validation_init_op = iterator.make_initializer(validation_dataset)

#Make the network
is_training_placeholder, output, loss = ConstructNetwork(images, labels, Parameters.IMAGE_SIZE, num_channels, Parameters.NUMBER_OF_NETWORK_OUTPUTS)

session.run(tf.global_variables_initializer())

#Add the optimizer
optimizer = AddOptimizer(output, loss)

session.run(tf.global_variables_initializer())

#Train the network with checkpointing and logging
saver = tf.train.Saver()


#numberOfEpochPerDataset, numberOfDatasets, checkpointPath, saver, outputOp, trainingOp, lossOp, dataIterator, validationDatasetInitOp, isTrainingPlaceHolder
Train(Parameters.NUMBER_OF_REPORT_CYCLES, Parameters.NUMBER_OF_DATA_GENERATION_CYCLES, './object_transform-model', saver, output, optimizer, loss, iterator, validation_init_op, is_training_placeholder, labels, image_names)




 


