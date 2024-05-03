########################################################################################################################
# Imports
########################################################################################################################
from skimage import io
import skimage as skimage
import AdditionalUtils.PrintFormat as PrintFormat
import os

########################################################################################################################
# Load image per frame camera
########################################################################################################################

def load_images_per_frame_camera(basePath, firstFrame, lastFrame, numCameras, ending, X=-1 , Y=-1 ):
    images = []
    for f in range(0, (lastFrame+1) - firstFrame):
        images.append([])
        for c in range(0, numCameras):

            imagePath = basePath + str(c) + "/image_c_"+ str(c) +"_f_" + str(firstFrame + f) + ending

            if not os.path.exists(imagePath):
                PrintFormat.printError('File: ' + imagePath + 'does not exist!')
                PrintFormat.printWarning('Try alternative image ending')
                if ending == '.jpg':
                    ending = '.png'
                    PrintFormat.printWarning('Switched from jpg to png')
                elif ending == '.png':
                    ending = '.jpg'
                    PrintFormat.printWarning('Switched from png to jpg')
                else:
                    PrintFormat.printError('Unsupported ending type!')
                    quit()
                imagePath = basePath + str(c) + "/image_c_" + str(c) + "_f_" + str(firstFrame + f) + ending

                if os.path.exists(imagePath):
                    PrintFormat.printWarning('Successfully found file with alternative ending --> Proceeding')
                else:
                    PrintFormat.printError('Still no image was found --> Return')
                    quit()

            img = io.imread(imagePath)
            if X > 0 and Y > 0:
                img = skimage.transform.resize(img,(Y,X))
            images[f].append(img)
    return images