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
cascPath = sys.argv[1]

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
    
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(1)

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
        


if __name__ == '__main__':
    Thread(target = Camera).start()
    Thread(target = Sound).start()
