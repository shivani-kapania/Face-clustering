#!/usr/bin/env python
# coding: utf-8



import paths
import face_recognition
import pickle
import cv2
import os


dataset_path = './dataset'
detection_method = 'hog'


imagePaths = list(paths.list_images(dataset_path))
data = []


#imagePaths


# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    print(imagePath)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
 
    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
        for (box, enc) in zip(boxes, encodings)]
    data.extend(d)


f = open('encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()




