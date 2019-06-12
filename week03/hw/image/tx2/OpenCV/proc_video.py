import numpy as np
import time
import cv2
import paho.mqtt.client as mqtt

MQTT_TOPIC = "_tx2/webcam/detected_faces"

def publish_faces(gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    print (len(faces))
    for face in faces:
        rc, jpg = cv2.imencode('.png', face)
        msg = jpg.tobytes()
        client1.publish(MQTT_TOPIC, msg)
        
if __name__ == '__main__':
    # Use pre-trained Face Cascade Classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 1 should correspond to /dev/video1 , USB camera. The 0 is reserved for the TX2 onboard camera
    cap = cv2.VideoCapture(1)    
    # set connection to broker
    broker = "mosquitto"
    port = 1883
    client1= mqtt.Client("detector1")
    client1.connect(broker, port)

    while(True):
        # Capture each frame
        ret, frame = cap.read()
        # getting rid of colors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face detection and other logic goes here
        publish_faces(gray)
        time.sleep(1)
