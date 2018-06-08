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

#cam = bpy.data.objects['Camera']
#obj = bpy.data.objects['Petshop-cat-figurine']

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
def GenerateExamples(numberOfExamples, objectName, cameraName, minDistance, maxDistance, locationNormalizationFactor, directoryPath, includeLocation, includeRotation):
    #Error on first render, so skip writing that one
    bpy.ops.render.render( write_still=False )

    results_map = dict()
    #Setup keyframes for animation to produce images
    bpy.data.scenes['Scene'].render.filepath = directoryPath + "/" + "example#"
    for example_index in range(0, numberOfExamples):
        #Set keyframe
        current_frame_number = example_index+1        
        bpy.context.scene.frame_set(current_frame_number)

        example_name = "example" + str(current_frame_number) + ".png"
        relative_vector, relative_euler_orientation = PerturbInsideCameraView(minDistance, maxDistance, bpy.data.objects[cameraName], bpy.data.objects[objectName])
        relative_matrix_orientation = relative_euler_orientation.to_matrix()
        bpy.data.objects[objectName].keyframe_insert(data_path='location', index=-1 )
        bpy.data.objects[objectName].keyframe_insert(data_path='rotation_euler', index=-1)
        bpy.data.objects[objectName].keyframe_insert(data_path='scale', index=-1)

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
    
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = numberOfExamples

    #convert to NLA tracks instead of actions because actions aren't working for some reason
    for ob in bpy.context.scene.objects:
        if ob.animation_data is not None:
            action = ob.animation_data.action
            if action is not None:
                track = ob.animation_data.nla_tracks.new()
                track.strips.new(action.name, action.frame_range[0], action)
                ob.animation_data.action = None

    bpy.context.scene.update()     
    bpy.ops.render.render( animation=True, write_still=True )

    #Store labels in JSON file
    json_string = json.dumps(results_map, sort_keys=True, indent=4)
    json_file = open(directoryPath+"/labels.json", "w")
    json_file.write(json_string)
    json_file.close()

start = time.time()
GenerateExamples(10000, "Petshop-cat-figurine", "Camera", 1.0, 2.0, 1.0, "/home/charlesrwest/storage/Datasets/objectTransform/rawData", Parameters.TRANSLATION_TRACKING_ENABLED, Parameters.ROTATION_TRACKING_ENABLED)
