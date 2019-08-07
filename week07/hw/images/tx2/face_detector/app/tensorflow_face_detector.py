import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from subprocess import Popen, PIPE


DETECTION_THRESHOLD = 0.5
FROZEN_GRAPH_NAME = '/tmp/frozen_inference_graph_face.pb'

## Download an SSD model for face detection
if not os.path.exists(FROZEN_GRAPH_NAME):
    process = Popen(f"wget https://github.com/yeephycho/tensorflow-face-detection/blob/master/model/frozen_inference_graph_face.pb?raw=true -O {FROZEN_GRAPH_NAME}",
                    shell=True, stdout=PIPE)
    process.wait()
    print (process.returncode)



class TfFaceDetector():
    def __init__(self, threshold=DETECTION_THRESHOLD):
        self.detection_threshold = threshold

        ## Load the frozen graph
        frozen_graph = tf.GraphDef()
        output_dir = ''
        with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
            frozen_graph.ParseFromString(f.read())

        ## A few magical constants
        INPUT_NAME='image_tensor'
        BOXES_NAME='detection_boxes'
        CLASSES_NAME='detection_classes'
        SCORES_NAME='detection_scores'
        MASKS_NAME='detection_masks'
        NUM_DETECTIONS_NAME='num_detections'

        input_names = [INPUT_NAME]
        output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

        ## Optimize the frozen graph using TensorRT
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50
        )

        ## Create session and load graph
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.tf_sess = tf.Session(config=tf_config)

        # use this if you want to try on the optimized TensorRT graph
        # Note that this will take a while
        # tf.import_graph_def(trt_graph, name='')

        # use this if you want to try directly on the frozen TF graph
        # this is much faster
        tf.import_graph_def(frozen_graph, name='')

        self.tf_input = self.tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
        self.tf_scores = self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
        self.tf_boxes = self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        self.tf_classes = self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
        self.tf_num_detections = self.tf_sess.graph.get_tensor_by_name('num_detections:0')


    def detect_face(self, image):

        image = np.array(image)

        ## Run network on Image
        scores, boxes, classes, num_detections = self.tf_sess.run(
            [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_num_detections], 
            feed_dict={
                self.tf_input: image[None, ...]
                })

        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])

        ## Display Results
        if any(scores > self.detection_threshold):
            return True
        else:
            return False