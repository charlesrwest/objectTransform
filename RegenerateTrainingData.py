from subprocess import call
import dataset
import convertToTfRecord

import os
import glob

def RemoveFilesInFolder(folderPath):
    files = glob.glob(folderPath + "/*")
    for f in files:
        os.remove(f)

def RegenerateTrainingData():
    RemoveFilesInFolder("/home/charlesrwest/cpp/Datasets/objectTransform/rawData")
    
    devnull = open(os.devnull, 'w')
    call(["/home/charlesrwest/cpp/libraries/blender-2.80-beta/blender", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransformWith2AxesOrientation/catRepositionedTest.blend", "--background", "--python", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransformWith2AxesOrientation/GenerateExampleData.py"], stdout=devnull, stderr=devnull)
#    call(["/home/charlesrwest/cpp/libraries/blender-2.80-beta/blender", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransformWith2AxesOrientation/catRepositionedTest.blend", "--background", "--python", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransformWith2AxesOrientation/GenerateExampleData.py"])

    convertToTfRecord.ConvertToTfRecord("/home/charlesrwest/cpp/Datasets/objectTransform/rawData", "/home/charlesrwest/cpp/Datasets/objectTransform/objectTransformDatasetTrain2Axis.tfrecords", 224, 224)

    RemoveFilesInFolder("/home/charlesrwest/cpp/Datasets/objectTransform/rawData")
