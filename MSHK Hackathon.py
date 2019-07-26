#!/usr/bin/env python
# coding: utf-8

# In[7]:


# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageGrab
import cv2
import requests
import json
from threading import Thread
import playsound
import argparse

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)

# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="alert.wav",help="path alarm .WAV file")
args = vars(ap.parse_args())


# In[6]:


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


# In[5]:


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
SCREEN_SIZE = [0, 50, 1400, 1100]
SCREEN_HEIGHT = 1100-50
SCREEN_WIDTH = 1400-0
ALARM_ON = False
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
      pil_im = ImageGrab.grab(SCREEN_SIZE)  # entire screen
      image_np = cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      filteredBoxes = []
      filteredClasses = []
      filteredScores = []
    
      squeezeBoxes = np.squeeze(boxes)
      squeezeClasses = np.squeeze(classes).astype(np.int32)
      squeezeScores = np.squeeze(scores)
      for i in range (0,len(squeezeClasses)):
          if squeezeClasses[i] == 1 or squeezeClasses[i] == 3 or squeezeClasses[i] == 6 or squeezeClasses[i] == 8:
            if (squeezeBoxes[i][3]*SCREEN_WIDTH - squeezeBoxes[i][1]*SCREEN_WIDTH) < (SCREEN_WIDTH * 0.9) and (squeezeBoxes[i][2]*SCREEN_HEIGHT - squeezeBoxes[i][0]*SCREEN_HEIGHT) < (SCREEN_HEIGHT * 0.9):
                filteredBoxes.append(squeezeBoxes[i])
                filteredClasses.append(squeezeClasses[i])
                filteredScores.append(squeezeScores[i])
    
      filteredBoxes = np.array(filteredBoxes)
      filteredClasses = np.array(filteredClasses)
      filteredScores = np.array(filteredScores)
    
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          filteredBoxes,
          filteredClasses,
          filteredScores,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8,
          agnostic_mode=False,
          min_score_thresh=.5)
      
      errorTypes = []
      objectClasses = []
      for i in range (0, len(filteredBoxes)):
          if filteredScores[i] >= 0.5:
            mid_x = (filteredBoxes[i][1]+filteredBoxes[i][3])/2
            mid_y = (filteredBoxes[i][0]+filteredBoxes[i][2])/2
            apx_distance = round(((1 - (filteredBoxes[i][3] - filteredBoxes[i][1]))**4),1)
            

            if apx_distance <=0.5 and apx_distance >=0.3:
              if mid_x > 0.3 and mid_x < 0.7:
                errorTypes.append('yellow')
                objectClasses.append(filteredClasses[i])
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*SCREEN_WIDTH),int(mid_y*SCREEN_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                            args=(args["alarm"],))
                        t.deamon = True
                        t.start()
            elif apx_distance <=0.3:
              if mid_x > 0.3 and mid_x < 0.7:
                errorTypes.append('red')
                objectClasses.append(filteredClasses[i])
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*SCREEN_WIDTH),int(mid_y*SCREEN_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image_np, 'DANGER!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                            args=(args["alarm"],))
                        t.deamon = True
                        t.start()
            else:
              cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*SCREEN_WIDTH),int(mid_y*SCREEN_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 51, 204), 2)
              errorTypes.append('green')
              objectClasses.append(filteredClasses[i])
              ALARM_ON = False
          else:
            errorTypes.append('green')
            objectClasses.append(filteredClasses[i])
            ALARM_ON = False
                
            
        
            
      url = 'http://localhost:3000/get-prediction-data'
      payload = {'errorTypes': errorTypes}
      headers = {'content-type': 'application/json'}

      response = requests.post(url, data=json.dumps(payload), headers=headers)
      try:
          response_data = response.raise_for_status()
      except:
          pass

      
      cv2.imshow('window',cv2.resize(image_np,(800,450)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break


# In[9]:


pil_im = ImageGrab.grab([52, 332, 1300, 1000])  # entire screen
pil_im.save('screenshot.png')


# In[ ]:




