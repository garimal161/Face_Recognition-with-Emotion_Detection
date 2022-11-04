# Face_Recognition-with-Emotion_Detection

This is a Facial-recognition project which will also classify Emotions

# Benefits - 
As we all know that many Machine-Learnig based project reqiures altest 2-3 similar kind of images to train & ofcourse a lot of time to train all the things BUT here it requires just one good* image of the person and thats all needed for recognition. 

# This Project Uses:-
Language --> Python=3.8 
System --> Nvidia Jetson-nano
Operating System --> Ubuntu
CSI-Camera (USB Camera is also Perfect)

# Prerequisite Libraries / Packages

1. face_recognition
install using - pip install face_recognition
import using - import face_recognition

2. cv2
install using - pip install opencv-python
import using - import cv2

3. os
install using - "Usually Preinstalled"
import using - import os

4. pandas
install using - pip install pandas
import using - import pandas as pd / import cvlib

5. datetime
install using - pip install datetime
import using - from datetime import datetime

6. tkinter
install using - pip install tinkter
import using - from tkinter import *

7. tesorflow 2.0 (If GPU is used Verson of CUDA can create exceptions, so try to install the same version on tensorflow_gpu)
 install using - pip install tensorflow (For GPU users use - pip install tensorflow_gpu
 import using - import tensorflow as tf
 
8. numpy
install using - pip install numpy
import using - import numpy as np

9. deepface
install using - pip install deepface
import using - from deepface import DeepFace

10. Harcascade (for Frontal face)
download using - github: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# Important Functuions used

1. "Emotion" : To detect Emotion of the person detected in the frame

2. "update" : This function is used to update face records

3. "facee" : Use to recognize face of the person in the frame
