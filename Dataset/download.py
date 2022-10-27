import csv
import gdown
import requests
import pandas as pd
from config import *

def create_dictionary ():
  d = {}  
  url = bsource + '/file/d/'+ csv_id + '/view?usp=sharing'
  url = bsource + '/uc?id=' + url.split('/')[-2]
  df = pd.read_csv (url, sep = ';')
  for index,row in df.iterrows():
    image_name = row[0]
    url = row[1]
    d[image_name] = url
  return d  
    
def download_image (name, url):
  gdown.download (url, name + '.jpg', quiet=False)

def download_dataset (base):
  for index,row in base.iterrows():
    image_id = row['ID']
    url = row['URL']
    with open(image_id + '.jpg', 'wb') as handle:
      status = requests.get (url, stream=True)
    if not status.ok or status.history:
      download_image (image_id, d[image_id])
    else:
      for block in response.iter_content(1024):
        if not block:
          break
        handle.write(block)

# Download dataset:
base = pd.read_csv ('images_sources.csv')
d = create_dictionary ()
download_dataset (base)

