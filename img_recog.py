import numpy as np
import cv2

import os
import shutil
from scipy.sparse.construct import random


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils



# ########
# cleaning and preprocessing of the data
# facedetection with the help of haarcascade
# cropped faces are automatically stored labelled directories
########

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')


#get_cropped_image_if_2_eyes returns an cropped image of the original image if it has more than two eyes
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes =eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) <=2:
            return roi_color

path_to_data = "./dataset/images"
path_to_crop_data = "./cropped_image/"

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

if os.path.exists(path_to_crop_data):
    shutil.rmtree(path_to_crop_data)
os.mkdir(path_to_crop_data)

cropped_image_dirs = []
cricketer_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    cricketer_name = img_dir.split('\\')[-1]
    print(cricketer_name)
    cricketer_file_names_dict[cricketer_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_crop_data + cricketer_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)
            cropped_file_name = cricketer_name + str(count) + ".png"
            cropped_file_path = cropped_folder +  "/" + cropped_file_name

            cv2.imwrite(cropped_file_path,roi_color)
            cricketer_file_names_dict[cricketer_name].append(cropped_file_path)
            count +=1

#######
# class_dict stores the cricketers names and the corresponding labels
#######
class_dict ={}
count = 0
for cricketer_name in cricketer_file_names_dict.keys():
    class_dict[cricketer_name] = count
    count += 1

#######
# we start by defining the X and y datasets
# later we use train_test_split to split the dataset into train and test dataset
#######
X,y=[],[]
for cricketer_name,training_files in cricketer_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue 
        scaled_raw_img = cv2.resize(img,(32,32))
        X.append(scaled_raw_img)
        y.append(class_dict[cricketer_name])

X = np.array(X).astype(float)
train_X, test_X, train_y, test_y = train_test_split(X,y,random_state = 0)
train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0

train_Y=np_utils.to_categorical(train_y)
test_Y=np_utils.to_categorical(test_y)
num_classes=test_Y.shape[1]



#######
#  we build the CNN model for our classifier
#  then layers have to be added
#  some parameters have to be tweaked here and there to get a good accuracy
#######
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'])
model.summary()


model.fit(train_X,train_Y,
    validation_data=(test_X,test_Y),
    epochs=100,batch_size=32)
_,acc=model.evaluate(test_X,test_Y)
print(acc*100)


######
# save the model
######
model.save("img_recog_50epoch.h5")
