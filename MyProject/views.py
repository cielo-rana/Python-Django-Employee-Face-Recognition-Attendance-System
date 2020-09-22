from django.shortcuts import render, redirect
from cv2 import cv2
import os, sys
import numpy as np
from PIL import Image
from django.contrib import messages
from UI.models import Employee, Attendance
import datetime


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



def create_dataset(request):
    # Get userid from input from form
    userId = request.POST['userId']
    # camture images from the webcam and process and detect the face
    vid_cam = cv2.VideoCapture(0)
    # Detect face with default train data
    face_detector = cv2.CascadeClassifier('media/haarcascade_frontalface_default.xml')
    # set in to 0 
    count = 0

    while(True):
        # cam.read will return the status variable and the captured colored image
        _, image_frame = vid_cam.read()
        # to convert into grayscale 
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        # to detect all the images in the current frame,
        # and to return the coordinates of the faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        # In above 'faces' variable there can be multiple faces 
        # so we have to get each and every face and draw a rectangle around it
        for (x,y,w,h) in faces:
            # the initial point of the rectangle will be x,y and
            # end point will be x+width and y+height
            # along with color of the rectangle
            # thickness of the rectangle
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            # Whenever the program captures the face, we will write that in a folder
            count += 1
            # saves image in dataset as User.1.2 for userId 1 and sample 2
            cv2.imwrite("media/dataset/User." + str(userId) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            # shows window name frame with image
            cv2.imshow('frame', image_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # capture 50 images
        elif count>=100:
            print("Successfully Captured")
            break
    # release the video
    vid_cam.release()
    # close all windows
    cv2.destroyAllWindows()

    return redirect('/register')



def trainer(request):


    # Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # train with the default data
    detector= cv2.CascadeClassifier("media/haarcascade_frontalface_default.xml")
   
    path = 'media/dataset'
    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths
        ids = []
        # Now, we loop all the images and store that userid and the face with different image list
        faceSamples = []
        
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            PIL_img = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            img_numpy = np.array(PIL_img, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            id = int(os.path.split(imagePath)[-1].split(".")[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            
            # extract the face from the training image sample
            faces=detector.detectMultiScale(img_numpy)
            # If a face is there then append that in the list as well as Id of it
            
            for (x,y,w,h) in faces:
                # images
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                # label
                ids.append(id)
            
            
        return faceSamples, ids

    # Fetching ids and faces
    faces, ids = getImagesWithID(path)
    

    #Training the recognizer
    # For that we need face samples and corresponding labels
    
    recognizer.train(faces, np.array(ids))

    # Save the recogzier state so that we can access it later
    recognizer.write('media/trainer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/register')


def detect(request):
    faceDetect = cv2.CascadeClassifier('media/haarcascade_frontalface_default.xml')

    
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create()
    # loading the training data
    rec.read('media/trainer/trainingData.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX
    # assign id = 0
    id = 0
    # creating name array
    #names = ['None', 'Akash', 'Sabin', 'Biplov', 'Dipesh']
    #fetching data from database
    em = Employee.objects.values_list('name', flat=True)

    names = list(em)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while(True):
        # cam.read will return the status variable and the captured colored image
        _,img = cam.read()
        img = cv2.flip(img, 1) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            id,conf = rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

             
            if (conf< 100):
                id = names[id]
                conf = "  {0}".format(round(100 - conf))
                
                # performs attendance only when we have 35 or more accuracy
                if (int(conf)> 35):
                    date = datetime.datetime.now().strftime("%Y-%m-%d")

                    old = Attendance.objects.filter(name=id,date__date=date).first()
    
                    if(old == None):
                        at = Attendance()
                        at.name = id
                        at.save()
                    
                else:
                    messages.info(request,'oops!! not detected!!')
            else:
                id = "unknown"
                conf = "  {0}".format(round(100 - conf))
        
            cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
            cv2.putText(
                    img, 
                    str(conf), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )

        cv2.imshow("Face",img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/attendance')