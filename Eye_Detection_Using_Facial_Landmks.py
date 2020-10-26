#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:12:56 2020

@author: Doaa
"""
#================================ Importing Libraries ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from random import randint
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dropout, MaxPooling2D

#============================ Data Preprocessing ==================================
faces = np.load('face_images.npz')
faces = faces['face_images']
faces = np.moveaxis(faces, -1, 0)

landmarks = pd.read_csv('facial_keypoints.csv')

prediction_fields = landmarks.iloc[:,[0,1,2,3]]

rows_with_null = [index for index, row in prediction_fields.iterrows() if row.isnull().any()]
rows_without_null = [index for index, row in prediction_fields.iterrows() if not(row.isnull().any())]

# Drop Rows with NAN values 
Y = (prediction_fields.drop(prediction_fields.index[rows_with_null])).to_numpy()

faces = faces[rows_without_null,:,:]

m = len(rows_without_null)
n = faces.shape[1]

X = faces.reshape(m,n,n,1)

#============================== Function to plot Training Image =======================
def plot_pretrained_image(img, keypoints, axis, title):
    
    #img = img.reshape(96,96)
    axis.imshow(img, cmap='gray')
    axis.scatter(keypoints[0::2], keypoints[1::2], marker='X',c='r',s=100)
    plt.title(title)

def plot_trained_image(img, keypoints, trained, axis, title):
    
    axis.imshow(img, cmap='gray')
    axis.scatter(keypoints[0::2], keypoints[1::2], c = 'r')
    axis.scatter(trained[0::2], trained[1::2], c = 'b')
    axis.set_xlabel(title)

# Example Images from Data before Training
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))
plt.setp(axes.flat, xticks = [], yticks = [])
for i, ax in enumerate(axes.flat):
    index = randint(0, 2260)
    title = 'Face_' + str(index)
    plot_pretrained_image(X[index], Y[index], ax, title)
plt.show()    

#============================== Data Dividing =======================
random_seed = 21

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

# Normaliza the input image
X_train = X_train / 255.0
X_test = X_test / 255.0


#============================== Building Model =======================
model_1 = Sequential([
    BatchNormalization(input_shape = (n, n, 1)),
    
    Conv2D(24, (5, 5), padding = 'same', activation = 'relu', input_shape = (n, n, 1)),
    MaxPooling2D(2, 2, padding = 'valid'),
    Dropout(rate = 0.75),
    
    Conv2D(36, (5, 5), activation = 'relu'),
    MaxPooling2D(2, 2, padding = 'valid'),
    Dropout(rate = 0.75),
    
    Conv2D(48, (5, 5), activation = 'relu'),
    MaxPooling2D(2, 2, padding = 'valid'),
    Dropout(rate = 0.75),
    
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2, padding = 'valid'),
    Dropout(rate = 0.75),
    
    Conv2D(64, (3, 3), activation = 'relu'),
    GlobalAveragePooling2D(),
    Dropout(rate = 0.75),
    
    Dense(500, activation = 'relu'),
    Dropout(rate = 0.75),
    Dense(90, activation = 'relu'),
    Dropout(rate = 0.1),
    Dense(4),
])

model_1.summary()

#============================== Compiling Model =======================

model_1.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])

history_3 = model_1.fit(
    X_train, y_train,
    validation_data = (
        X_test,
        y_test
    ),
    batch_size = 20,
    epochs = 5,
    shuffle = True,
    verbose = 1
)

model = load_model('model_1_Eye.h5')
model_1.evaluate(X_test, y_test, verbose = 1)
#model_1.save('model_1_Eye.h5')

#============================== Plotting Test Images =======================
def plot_keypoints(img_path, face_cascade_path, model_path, scale=1.1, neighbors=6, key_size=10):
    
    face_cascade=cv2.CascadeClassifier(face_cascade_path) 
    model = load_model(model_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, neighbors)
    #fig = plt.figure(figsize=(25, 25))
    #ax = fig.add_subplot(121, xticks=[], yticks=[])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title('Image with Facial Keypoints')

    # Print the number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(img)

    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
        bgr_crop = image_with_detections[y:y+h, x:x+w] 
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255.0
        
        landmarks = model.predict(resize_gray_crop.reshape(1,96,96,1))[0]
        ax.scatter(((landmarks[0::2] )*orig_shape_crop[0]/96)+x, 
                   ((landmarks[1::2] )*orig_shape_crop[1]/96)+y, 
                   marker='o', c='r', s=key_size)
        
    ax.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))


# Paint the predicted keypoints on the test image
doaa = plot_keypoints('obamas4.png',
                        'haarcascade_frontalface_default.xml',
                        'model_1_Eye.h5')


