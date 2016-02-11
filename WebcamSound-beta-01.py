import threading
from threading import Thread
import cv2
import sys
#import wikipedia
#from chatterbot import ChatBot 
import shlex, subprocess 
import speech_recognition as sr
import pyvona
from googlesearch import GoogleSearch
import xml.etree.ElementTree as ET
import requests
import os 
from PIL import Image 
#cascPath = sys.argv[1]
import numpy as np

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer() 


def wikileaks(string):
    string=wikipedia.summary(string,sentences=1)
    chatvoice(string)
def speak():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source) # listen for 1 second to calibrate the energy threshold for ambient noise levels
        print("Say something!")
        audio = r.listen(source)

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        string =  r.recognize_google(audio)
        print "you said "+string
        return string 
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def Google1(string):
    gs = GoogleSearch(string)
    for hit in gs.top_results():
        #send(hit[u'content'])
        chatvoice(hit[u'content'])
        break


def chatvoice(string):
    
    v = pyvona.create_voice('username','password')
    #v.region('en-IN')
    #print v.list_voices() 
    v.speak(string)
    #v.speak(a)     



        
def intelbot(string):
    payload = {'input':string,'botid':'9fa364f2fe345a10'}
    r = requests.get("http://fiddle.pandorabots.com/pandora/talk-xml", params=payload)
    for child in ET.fromstring(r.text):
        if child.tag == "that":
            chatvoice(child.text)


def Camera():
    
    #faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            flags = 0
        )   
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # if cv2.waitKey(1) & 0xFF == ord('f'):
            #     roi = gray[y:y+h, x:x+w]
            #     cv2.imwrite('rahul.sad.png',roi)

        # Display the resulting frame
        cv2.imshow('Video', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def Sound():
    while True:
        takeString = speak()
        intelbot(takeString)
        
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels


if __name__ == '__main__':
    # Path to the Yale Dataset
    a = 1
    if a:
        path = './yalefaces'
        # Call the get_images_and_labels function and get the face images and the 
        # corresponding labels
        images, labels = get_images_and_labels(path)
        cv2.destroyAllWindows()

        # Perform the tranining
        recognizer.train(images, np.array(labels))

        # Append the images with the extension .sad into image_paths
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
        for image_path in image_paths:
            predict_image_pil = Image.open(image_path).convert('L')
            predict_image = np.array(predict_image_pil, 'uint8')
            faces = faceCascade.detectMultiScale(predict_image)
            for (x, y, w, h) in faces:
                nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
                nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
                if nbr_actual == nbr_predicted:
                    print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                else:
                    print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
                cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
                cv2.waitKey(1000)

    #a=Thread(target = Camera).start()
    #b=Thread(target = Sound).start()
