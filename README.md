# Cricketer-Image-Classification-using-keras

Using an image classification model to find out which cricketer is there in the input image.
LEVEL 1
Create a classifier, to find out the name of the cricketer from given image.
LEVEL 2
Write a python script to turn on the webcam, and put a square around the face and show which 
cricketer does that person look like


1:- the dataset was downloaded from kaggle
    the images in the dataset were then converted to cropped images or faces were extracted from the images.
    haarcascade was used to detect the faces in images and webcam stream
    confusing and irrelevant images were identified and deleted manually

2:- for the classification model, a CNN (Convolutional neural networks) was implemented
    KERAS library package was used for the that purpose
    Build a model architecture (Sequential) with Dense layers
    Train the model and make predictions
    
3:- Built a GUI using Tkinter. Added few buttons like "upload image" and "start webam"
    The "upload image" button leads to a dialog box to select a image to classify.
    It also activates the "classify" button to run the classification model for the uploaded image
    The "start webcam" button starts the webcam and detects the face/faces from the stream
    It then puts a green rectangle around the detected faces and classifies the image inside the rectangle
    The result is shown along with the rectangle at real time
    

