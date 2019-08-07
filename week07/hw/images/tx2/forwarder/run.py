import time
import paho.mqtt.client as mqtt

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
    # print("Connected with result code "+str(rc))
    client.subscribe("_tx2/webcam/#", qos=0)

def on_message(client, userdata, msg):    
    client_pub.publish(msg.topic, msg.payload, qos=1)
    print ("message published")


client_sub = mqtt.Client("subscriber")
client_pub = mqtt.Client("publisher")

client_sub.on_connect = on_connect
client_sub.on_message = on_message

client_sub.connect("mosquitto", 1883, 10)
client_pub.connect("169.45.94.22", 1883, 20)
client_sub.loop_forever()