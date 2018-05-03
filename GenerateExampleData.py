import bpy
import bmesh
from mathutils import Vector
from mathutils import Euler
from bpy_extras.object_utils import world_to_camera_view
import random
import math
import json

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
        relative_vector = GetRandomUnitVector()*random.uniform(minCenterDistance, maxCenterDistance)
        bpy.context.scene.update()
        obj.location = camera.location + relative_vector
        obj.rotation_euler = relative_euler_orientation

        #Update world matrix of object to match
        bpy.context.scene.update() 

        if not object_center_in_camera_view(camera, obj):
            continue

        if mesh_object_vertices_in_camera_view(camera, obj):
            break
    return relative_vector, relative_euler_orientation

#Multiplies the location of object by the normalization factor before storing it in JSON
#Currently requires camera to be at origin without rotation
def GenerateExamplesWithLocationAndOrientation(numberOfExamples, objectName, cameraName, minDistance, maxDistance, locationNormalizationFactor, directoryPath):

    results_map = dict()
    for example_index in range(0, numberOfExamples):
        example_name = "example" + str(example_index) + ".png"
        relative_vector, relative_euler_orientation = PerturbInsideCameraView(minDistance, maxDistance, bpy.data.objects[cameraName], bpy.data.objects[objectName])
        relative_matrix_orientation = relative_euler_orientation.to_matrix()

        #Render image
        bpy.data.scenes['Scene'].render.filepath = directoryPath + "/" + example_name
        bpy.ops.render.render( write_still=True )

        #Store data for label (location, X axis, Y axis)
        results_map[example_name] = (relative_vector.x*locationNormalizationFactor, relative_vector.y*locationNormalizationFactor, relative_vector.z*locationNormalizationFactor, relative_matrix_orientation[0][0], relative_matrix_orientation[1][0], relative_matrix_orientation[2][0], relative_matrix_orientation[0][1], relative_matrix_orientation[1][1], relative_matrix_orientation[2][1])

    #Store labels in JSON file
    json_string = json.dumps(results_map, sort_keys=True)
    json_file = open(directoryPath+"/labels.json", "w")
    json_file.write(json_string)
    json_file.close()


#Currently requires camera to be at origin without rotation
def GenerateExamplesWithOrientation(numberOfExamples, objectName, cameraName, minDistance, maxDistance, directoryPath):
    #Do a quick render to wake up the HDR lighting (first image coming out without lights for some reason)
    bpy.ops.render.render( write_still=False )

    results_map = dict()
    for example_index in range(0, numberOfExamples):
        example_name = "example" + str(example_index) + ".png"
        relative_vector, relative_euler_orientation = PerturbInsideCameraView(minDistance, maxDistance, bpy.data.objects[cameraName], bpy.data.objects[objectName])
        relative_matrix_orientation = relative_euler_orientation.to_matrix()

        #Render image
        bpy.data.scenes['Scene'].render.filepath = directoryPath + "/" + example_name
        bpy.ops.render.render( write_still=True )

        #Store data for label (X axis, Y axis)
        results_map[example_name] = (relative_matrix_orientation[0][0], relative_matrix_orientation[1][0], relative_matrix_orientation[2][0], relative_matrix_orientation[0][1], relative_matrix_orientation[1][1], relative_matrix_orientation[2][1])

    #Store labels in JSON file
    json_string = json.dumps(results_map, sort_keys=True)
    json_file = open(directoryPath+"/labels.json", "w")
    json_file.write(json_string)
    json_file.close()

GenerateExamplesWithOrientation(10000, "Petshop-cat-figurine", "Camera", 1.0, 5.0, "/home/charlesrwest/cpp/Datasets/objectTransform/rawData")
