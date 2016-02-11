import wikipedia
import sys
# NOTE: this example requires PyAudio because it uses the Microphone class
from chatterbot import ChatBot 
import shlex, subprocess 
import speech_recognition as sr
import pyvona
from googlesearch import GoogleSearch
import xml.etree.ElementTree as ET
import requests

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
	
if __name__=="__main__":
	chatbot = ChatBot("Ron Obvious")

	# Train based on the english corpus
	chatbot.train("chatterbot.corpus.english")
	while True:
		takeString = speak()
		intelbot(takeString) 
		#chatvoice(chatbot.get_response(takeString))
		#Google1(takeString)
		#wikileaks(takeString)





	


	
