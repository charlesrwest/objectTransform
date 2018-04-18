from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import json

def LoadImage(imagePath, targetWidth, targetHeight):
    #Read and resize image
    #Opencv2 load images as BGR, convert it to RGB
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    return img

def ConvertToFloatListFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def ConvertToBytesFeature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

shuffle_data = True
folder_path = sys.argv[1]
output_filename = sys.argv[2]
target_width = int(sys.argv[3])
target_height = int(sys.argv[4])

#Load the json file which has all of the image names and the associated output values
json_path = folder_path + "/labels.json"

labels = json.load(open(json_path))

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(output_filename)

#Load the associated images and save them with the given expected output

i = 0
for file_name, expected_output in labels.iteritems():
    try:
        # print how many images are saved every 1000 images
        if not i % 1000:
            print 'Train data: {}/{}'.format(i, len(labels))
            sys.stdout.flush()

        # Load the image
        file_path = folder_path + "/" + file_name
        img = LoadImage(file_path, target_width, target_height)
        # Create a feature
        feature = {'expected_output': ConvertToFloatListFeature(expected_output),
               'image': ConvertToBytesFeature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        i += 1
    except:
        print("Skipping due to problem with image")

writer.close()
sys.stdout.flush()


