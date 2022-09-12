import json
import os
from pathlib import Path
import cv2
import requests
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import math
import numpy.matlib as npm

fa = "/content/train/labels.json" # path to train/test/val labels.json
with open(fa) as f:
    data = json.load(f)
names = []

for x in data['images']:
    names.append(x['file_name'].split('.')[0])

for name in names:
  id = 0
  h = 0
  w = 0
  for x in data['images']:
    if x['file_name'] == name + ".PNG":
      id = x['id']
      h = x['height']
      w = x['width']



  bbox = {}
  bbox['category'] = []
  bbox['x'] = []
  bbox['y'] = []
  bbox['w'] = []
  bbox['h'] = []
  for x in data['annotations']:
    if x['image_id'] == id:
      bbox['category'].append(x['category_id'])
      bbox['x'].append(x['bbox'][0])
      bbox['y'].append(x['bbox'][1])
      bbox['w'].append(x['bbox'][2])
      bbox['h'].append(x['bbox'][3])
  with open("/content/Darknet_format/test/labels/" + name + ".txt", 'w') as f: # Path to train/test/val Darknet format

    for i in range(len(bbox['x'])):
      if bbox['category'][i] != 0:
        f.write(str(bbox['category'][i] - 1) + " " + str((bbox['x'][i] + bbox['w'][i] / 2) / w)[:8] + " " + str((bbox['y'][i] + bbox['h'][i] / 2) / h)[:8] + " " +  str(bbox['w'][i] / w)[:8] + " " +  str(bbox['h'][i] / h)[:8])
        if i != len(bbox['x']) - 1:
          f.write("\n")

# You can use Roboflow instead