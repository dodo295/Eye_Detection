# Eye_Detection
The objective of this project is to predict Eye positions on face images and videos 

# Objective

--> The objective of this code is to predict Eye positions on face images. 
--> This can be used as a building block in several applications, such as: tracking Eye in images and video.

Dataset:
This is a Keypoint Face Detection dataset from Kaggle uploaded to allow the kernel to work on it.
https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points

--> The dataset contains 7049 facial images and up to 15 keypoints marked on them.

--> The keypoints are in the facialkeypoints.csv file. The image are in the faceimages.npz file.

--> The input image is given in the last field of the data files, and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels.


# Output

--> The output is a file saved in 'model_1_Eye. h5' and reach accurecy (0.9936034083366394)


# How to run

--> Install numpy, opencv, keras
--> type this command in your terminal: 'python3 main.py'

AND you can build your model using 'Eye_Detection_Using_Facial_Landmks.py'


# Examples
. Detect all faces using Haar Cascade Classifiers using OpenCV
. Detect Center Eye keypoint with a Convolutional Neural Network


