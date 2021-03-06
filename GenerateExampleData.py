import datetime
import sys
sys.path.insert(0, "/home/charlesrwest/cpp/projects/TensorFlow/objectTransform")

import bpy
import bmesh
from mathutils import Vector
from mathutils import Euler
from bpy_extras.object_utils import world_to_camera_view
import random
import math
import json

import time
import Parameters
import os

#cam = bpy.data.objects['Camera']
#obj = bpy.data.objects['Petshop-cat-figurine']

def RemoveImageFromMemory (passedName):
    # Extra test because this can crash Blender.
    img = bpy.data.images[passedName]
    try:
        img.user_clear()
        can_continue = True
    except:
        can_continue = False
    
    if can_continue == True:
        try:
            bpy.data.images.remove(img, True)
            result = True
        except:
            result = False
    else:
        result = False
    return result

def CollectBackgroundImagePaths(sourceDirectoryPath, imageExtensions):
    file_paths = []

    for root, dirs, files in os.walk(sourceDirectoryPath):
        for file_name in files:
            for extension in imageExtensions:
                if file_name.endswith(extension):
                    file_paths.append(root + "/" + file_name)
    return file_paths

def SetBackgroundImage(backgroundImagePath):
    #Store reference to old image so it can be removed
    old_image_name = bpy.data.materials[1].node_tree.nodes.get("Image Texture").image.name

    #Load new image to be used
    new_image = bpy.data.images.load(filepath = backgroundImagePath)

    #Set the new image to be used
    bpy.data.materials[1].node_tree.nodes.get("Image Texture").image = new_image

    #Delete the old one from the file
    remove_succeeded = RemoveImageFromMemory(old_image_name)

    if not remove_succeeded:
        print("Error removing " + old_image_name)
    else:
        print("Removing " + old_image_name + " worked")
    #bpy.data.images[old_image_name].user_clear()
    #bpy.data.images.remove(bpy.data.images[old_image_name])


def object_center_in_camera_view(camera, obj):
    scene = bpy.context.scene

    mat_world = obj.matrix_world
    cs, ce = camera.data.clip_start, camera.data.clip_end

    in_view = False

    co_ndc = world_to_camera_view(scene, camera, obj.location)
    in_view = (0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and cs < co_ndc.z <  ce)

    #print("Center view: " + str(in_view) + " with loc: " + str(obj.location) + " and camera rel position: " + str(Vector((co_ndc.x, co_ndc.y, co_ndc.z))))

    return in_view

def location_in_camera_view(camera, location):
    scene = bpy.context.scene

    cs, ce = camera.data.clip_start, camera.data.clip_end

    in_view = False

    co_ndc = world_to_camera_view(scene, camera, location)
    in_view = (0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and cs < co_ndc.z <  ce)
    
    #print("Center view: " + str(in_view) + " with loc: " + str(obj.location) + " and camera rel position: " + str(Vector((co_ndc.x, co_ndc.y, co_ndc.z))))

    return in_view

def approx_object_in_camera_view(camera, centerLocation, marginFactor, distance, minDistance, maxDistance):
    scene = bpy.context.scene

    adjusted_margin = marginFactor*(maxDistance/distance)

    cs, ce = camera.data.clip_start, camera.data.clip_end

    in_view = False

    co_ndc = world_to_camera_view(scene, camera, centerLocation)
    in_view = (adjusted_margin < co_ndc.x < (1.0-adjusted_margin) and adjusted_margin < co_ndc.y < (1.0-adjusted_margin) and cs < co_ndc.z <  ce)
    
    #print("Center view: " + str(in_view) + " with loc: " + str(obj.location) + " and camera rel position: " + str(Vector((co_ndc.x, co_ndc.y, co_ndc.z))))

    return in_view

def mesh_object_vertices_in_camera_view(camera, obj):
    scene = bpy.context.scene

    mat_world = obj.matrix_world
    cs, ce = camera.data.clip_start, camera.data.clip_end

    for vertex in obj.data.vertices:
        co_ndc = world_to_camera_view(scene, camera, mat_world * vertex.co)
        if(not (0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and cs < co_ndc.z <  ce)):
            #print("Vertex view: " + str(False))
            return False

    #print("Vertex view: " + str(True))
    return True

def GetRandomUnitVector():
    unit_vector = Vector((random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)))
    unit_vector.normalize()
    return unit_vector

def PerturbInsideCameraView(minCenterDistance, maxCenterDistance, camera, obj):
    relative_vector = Vector()
    relative_euler_orientation = Euler()    
    relative_euler_orientation.x = random.uniform(-math.pi, math.pi)
    relative_euler_orientation.y = random.uniform(-math.pi, math.pi)
    relative_euler_orientation.z = random.uniform(-math.pi, math.pi)

    #Find a position that is fully inside the camera's frustum
    obj.rotation_mode = 'XYZ'
    for i in range(0, 10000):
        distance = random.uniform(minCenterDistance, maxCenterDistance)
        relative_vector = GetRandomUnitVector()*distance

        if not approx_object_in_camera_view(camera, camera.location + relative_vector,  .15, distance, minCenterDistance, maxCenterDistance):
            continue

        bpy.context.scene.update()
        #print("Updating scene  " + str((time.time() - start )*1000.0) + " milliseconds")

        #Temporarily disabling translation for debugging
        obj.location = camera.location + relative_vector
        obj.rotation_euler = relative_euler_orientation

        #Update world matrix of object to match
        bpy.context.scene.update() 

        #if not object_center_in_camera_view(camera, obj):
        #    continue

        break
        #start = time.time()
        #if mesh_object_vertices_in_camera_view(camera, obj):
        #    print("Fine grain check  " + str((time.time() - start )*1000.0) + " milliseconds")
        #    break
       # print("Fine grain check  " + str((time.time() - start )*1000.0) + " milliseconds")
    return relative_vector, relative_euler_orientation

#Multiplies the location of object by the normalization factor before storing it in JSON
#Currently requires camera to be at origin without rotation
def GenerateExamples(numberOfExamples, objectName, cameraName, minDistance, maxDistance, locationNormalizationFactor, backgroundImagesDirectoryPath, outputDirectoryPath, includeLocation, includeRotation):
    background_image_paths = CollectBackgroundImagePaths(backgroundImagesDirectoryPath, [".png", ".jpg"])

    #Error on first render, so skip writing that one
    bpy.ops.render.render( write_still=False )

    results_map = dict()
    for example_index in range(0, numberOfExamples):
        current_frame_number = example_index+1 

#        if (example_index % 5) == 0:
#            bpy.ops.wm.save_as_mainfile(filepath="/home/charlesrwest/Downloads/test_blend.blend")

        if (example_index % 1000) == 0:
            print("Generated " + str(example_index) + " images at " + datetime.datetime.now().strftime("%I:%M%p:%S on %B %d, %Y"))

        if (example_index % 100000) == 0:
            #Store labels in JSON file
#            json_string = json.dumps(results_map, sort_keys=True, indent=4)
            json_string = json.dumps(results_map)
            json_file = open(outputDirectoryPath+"/labels.json", "w")
            json_file.write(json_string)
            json_file.close()

        example_name = "example" + str(current_frame_number) + ".png"

        while True:
            try:
                random_image_path = random.choice(background_image_paths)
                SetBackgroundImage(random_image_path)
                break
            except:
                pass
        
        relative_vector, relative_euler_orientation = PerturbInsideCameraView(minDistance, maxDistance, bpy.data.objects[cameraName], bpy.data.objects[objectName])
        relative_matrix_orientation = relative_euler_orientation.to_matrix()
        
        #Render image
        bpy.data.scenes['Scene'].render.filepath = outputDirectoryPath + "/" + example_name
        bpy.ops.render.render( write_still=True )

        #Store data for label (location, X axis, Y axis)
        output_list = []
        
        if(includeLocation):
            output_list.append(relative_vector.x*locationNormalizationFactor)
            output_list.append(relative_vector.y*locationNormalizationFactor)
            output_list.append(relative_vector.z*locationNormalizationFactor)

        if(includeRotation):
            output_list.append(relative_matrix_orientation[0][0])
            output_list.append(relative_matrix_orientation[1][0])
            output_list.append(relative_matrix_orientation[2][0])
            output_list.append(relative_matrix_orientation[0][1])
            output_list.append(relative_matrix_orientation[1][1])
            output_list.append(relative_matrix_orientation[2][1])

        results_map[example_name] = output_list
        print("Rendered " + example_name)

    #Store labels in JSON file
    #json_string = json.dumps(results_map, sort_keys=True, indent=4)
    json_string = json.dumps(results_map)
    json_file = open(outputDirectoryPath+"/labels.json", "w")
    json_file.write(json_string)
    json_file.close()

start = time.time()
GenerateExamples(Parameters.EXAMPLES_PER_EPOC, "Petshop-cat-figurine", "Camera", Parameters.MIN_OBJECT_CENTER_DISTANCE, Parameters.MAX_OBJECT_CENTER_DISTANCE, 1.0, "/home/charlesrwest/storage/Datasets/backgrounds/downloads", "/home/charlesrwest/storage/Datasets/objectTransform/rawData", Parameters.TRANSLATION_TRACKING_ENABLED, Parameters.ROTATION_TRACKING_ENABLED)
