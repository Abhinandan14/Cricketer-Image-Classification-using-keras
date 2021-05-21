
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


#######
# load the trained model to classify the images
#######
from keras.models import load_model
from img_recog import *
model = load_model("img_recog_50epoch.h5")


######
#initialise GUI
######
top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

#######
# the functionality of the classify button is defined here
#######
def classify(file_path):
    global label_packed
    im = get_cropped_image_if_2_eyes(file_path)
    key_list = list(class_dict.keys())
    val_list = list(class_dict.values())
    im=cv2.resize(im,(32,32))
    im=np.expand_dims(im,axis=0)
    im=np.array(im)
    pred=model.predict_classes([im])[0]
    label.configure(foreground='#011638', text=key_list[val_list.index(pred)]) 

    
#######
# show the classify button only when the upload image button is pressed
#######
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

#######
# to upload the image from the browsed path
#######
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
        (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

#######
# start the webcam for realtime image recognition
#######
def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
       _, img = cap.read()
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)
       
       for(x,y,w,h) in faces:
          roi_gray = gray[y:y+h, x:x+w]
          im = img[y:y+h, x:x+w]
          key_list = list(class_dict.keys())
          val_list = list(class_dict.values())
          im=cv2.resize(im,(32,32))
          im=np.expand_dims(im,axis=0)
          im=np.array(im)
          pred=model.predict_classes([im])[0]
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          cv2.putText(img, key_list[val_list.index(pred)], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
             
       cv2.imshow('img', img)
       ######
       # press 'q' to exit the webcam
       ######
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()

upload=Button(top,text="Upload an image",command=upload_image,
  padx=10,pady=5)
upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
start_webcam=Button(top,text="Start Webcam",command=start_webcam,
  padx=10,pady=5)
start_webcam.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=20)
start_webcam.pack(side=BOTTOM,pady=10)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

