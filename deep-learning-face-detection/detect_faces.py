# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt
# --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os,sys
import glob
import imutils

faces_path = "faces"
ext = '*.jpg'
face_width = 80
def extract_faces(image , count = 0):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            #y = startY - 10 if startY - 10 > 10 else startY + 10        
            #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            #(0, 0, 255), 2)
            #print(startY,endY,startX,endX)
            try:
                crop_img = image[startY:endY-1, startX:endX-1]
                crop_img = imutils.resize(crop_img, width=face_width)
                # write the image to disk with frame-count
                cv2.imwrite(  faces_path + "\img%d.jpg" % count, crop_img)            
            
                print("writing file: img%d.jpg" % count)
                count +=1
            except Exception:
                pass

    #cv2.imshow("Output", image)
    #cv2.imshow("cropped", crop_img)
    #cv2.waitKey(0)
    return count

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=None,
    help="path to input image")
ap.add_argument("-pr", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-p", "--imagepath", required=True,
    help="path to folder with images")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it

path = args["imagepath"]
faces_path = path + '/'+ faces_path
if not os.path.exists(faces_path): 
    os.mkdir(faces_path)
    print("Path %s is created" % (faces_path) )

path += '/'+ ext
print("Path to search faces: " + path)

count  =  sum(os.path.isfile(f) for f in glob.glob(faces_path+'/'+ ext))
print("Total %d files with extention %s found in %s" % (count,ext,path))

if args["image"] is not None:  
    image = cv2.imread(args["image"])
    extract_faces(image,count)
else:
    files = glob.glob(path)
    for file in files:
        print("file:" + file)
        image = cv2.imread(file)
        count = extract_faces(image,count)


