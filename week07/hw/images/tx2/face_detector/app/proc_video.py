import numpy as np
import cv2
import time
import paho.mqtt.client as mqtt
from tensorflow_face_detector import TfFaceDetector


MQTT_TOPIC = "_tx2/webcam/detected_faces"


def publish_image(image):
    rc, jpg = cv2.imencode('.png', image)
    msg = jpg.tobytes()
    client1.publish(MQTT_TOPIC, msg)


if __name__ == '__main__':

    # 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
    detector = TfFaceDetector()
    cap = cv2.VideoCapture(1)

    # set up connection to broker
    broker = "mosquitto"
    port = 1883
    client1= mqtt.Client("detector1")
    client1.connect(broker, port)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image_resized = cv2.resize(image, (300, 300))

        # face detection and other logic goes here
        is_face = detector.detect_face(image_resized)
        if is_face:
            publish_image(image)
        time.sleep(1)