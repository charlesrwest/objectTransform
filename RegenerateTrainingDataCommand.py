from sys import argv
import RegenerateTrainingData

#objectTransformDatasetTrain.tfrecords
#objectTransformDatasetValidate.tfrecords

RegenerateTrainingData.RegenerateTrainingData(argv[1])
