from subprocess import call
import dataset
import convertToTfRecord

import os
import glob
import Parameters

def RemoveFilesInFolder(folderPath):
    files = glob.glob(folderPath + "/*")
    for f in files:
        os.remove(f)

def RegenerateTrainingData(outputFileName):
    RemoveFilesInFolder("/home/charlesrwest/storage/Datasets/objectTransform/rawData")
    
    devnull = open(os.devnull, 'w')
#    call(["/home/charlesrwest/cpp/libraries/blender-2.80-beta/blender", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransform/catRepositionedTest.blend", "--background", "--python", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransform/GenerateExampleData.py"], stdout=devnull, stderr=devnull)
    call(["/home/charlesrwest/cpp/libraries/blender-2.80-beta/blender", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransform/catRepositionedTest.blend", "--background", "--python", "/home/charlesrwest/cpp/projects/TensorFlow/objectTransform/GenerateExampleData.py"])

    convertToTfRecord.ConvertToTfRecord("/home/charlesrwest/storage/Datasets/objectTransform/rawData", "/home/charlesrwest/storage/Datasets/objectTransform/" + outputFileName, Parameters.IMAGE_SIZE, Parameters.IMAGE_SIZE)

    RemoveFilesInFolder("/home/charlesrwest/storage/Datasets/objectTransform/rawData")


