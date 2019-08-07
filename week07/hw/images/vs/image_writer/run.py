import datetime
import time
import paho.mqtt.client as mqtt
import io
from PIL import Image

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
    # print("Connected with result code "+str(rc))
    client.subscribe("_tx2/webcam/#", qos=0)

def on_message(client, userdata, msg):
    # fn = f"detected_face_{datetime.datetime.now()}"
    # with open(f"/mnt/mybucket/hw7/{fn}", 'wb+') as f:
        # f.write(msg.payload)

    stream = io.BytesIO(msg.payload)
    img = Image.open(stream)
    fn = f"detected_face_{datetime.datetime.now()}.png"
    img.save(f"/mnt/mybucket/hw7/{fn}")
    print ("message saved")


client_sub = mqtt.Client()
client_sub.on_connect = on_connect
client_sub.on_message = on_message

client_sub.connect("mosquitto",1883, 10)
client_sub.loop_forever()