import json
import os
from pathlib import Path

with open('./test/labels.json') as f:
  data = json.load(f)
  for x in data['images']:
    old_name_images = './test/images/' + str(x['file_name'])
    old_name_labels = './test/labels/' + str(x['file_name'][:-4]) + '.txt'


    x['file_name'] = str(x['id']) + '.PNG'

    new_name_images = './new_test/images/' + str(x['file_name'])
    new_name_labels = './new_test/labels/' + str(x['file_name'][:-4]) + '.txt'


    os.rename(old_name_images, new_name_images)
    os.rename(old_name_labels, new_name_labels)

with open("./new_test/labels.json", "w") as outfile:
    json.dump(data, outfile)

