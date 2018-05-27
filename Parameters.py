TRANSLATION_TRACKING_ENABLED = True
ROTATION_TRACKING_ENABLED = True

NUMBER_OF_NETWORK_OUTPUTS = 0

if TRANSLATION_TRACKING_ENABLED : 
    NUMBER_OF_NETWORK_OUTPUTS += 3

if ROTATION_TRACKING_ENABLED :
    NUMBER_OF_NETWORK_OUTPUTS += 6


INITIAL_TRAINING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 224 #Need to update image generation to set image size according to this parameter
