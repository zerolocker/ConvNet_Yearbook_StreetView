from __future__ import print_function
from os import path
from math import sin, cos, atan2, sqrt, pi

from src.util import *


def numToRadians(x):
  return x / 180.0 * pi

# Calculate distance (km) between Latitude/Longitude points
# Reference: http://www.movable-type.co.uk/scripts/latlong.html
EARTH_RADIUS = 6371
def dist(lat1, lon1, lat2, lon2):
  lat1 = numToRadians(lat1)
  lon1 = numToRadians(lon1)
  lat2 = numToRadians(lat2)
  lon2 = numToRadians(lon2)

  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0)
  c = 2 * atan2(sqrt(a), sqrt(1-a))

  d = EARTH_RADIUS * c
  return d


# Evaluate L1 distance on valid data
def evaluateYearbook(Predictor):
  test_list = listYearbook(False, True)
  predictor = Predictor()

  total_count = len(test_list)
  l1_dist = 0
  print( "Total testing data", total_count )
  for image_name in test_list:
    image_path = path.join(YEARBOOK_PATH, image_name)
    pred_year = predictor.predict(image_path)
    truth_year = label(image_name)
    l1_dist += abs(pred_year - truth_year)
  l1_dist /= total_count
  print( "L1 distance", l1_dist )
  return l1_dist

# Evaluate L1 distance on valid data
def evaluateStreetview(Predictor):
  test_list = listStreetView(False, True)
  predictor = Predictor()

  total_count = len(test_list)
  l1_dist = 0
  print( "Total testing data", total_count )
  for image_name in test_list:
    image_path = path.join(STREETVIEW_PATH, image_name)
    pred_lat, pred_lon = predictor.predict(image_path)
    truth_lat, truth_lon = label(image_name)
    l1_dist += dist(pred_lat, pred_lon, truth_lat, truth_lon)
  l1_dist /= total_count
  print( "L1 distance", l1_dist )
  return l1_dist

if __name__ == "__main__":
  import importlib
  from argparse import ArgumentParser
  parser = ArgumentParser("Evaluate a model on the validation set")
  parser.add_argument("model_dir", help="The directory the model is stored in. This directory needs a run.py file inside it! Do not use any slashes in the path name.")
  args = parser.parse_args()
  try:
    run = importlib.import_module(args.model_dir+'.run')
  except:
    print("Failed to load 'run.py' in '%s'"%args.model_dir)
    exit(1)
  Predictor = run.Predictor
  if Predictor.DATASET_TYPE == 'yearbook':
    print("Yearbook")
    evaluateYearbook(Predictor)
  elif Predictor.DATASET_TYPE == 'streetview':
    print("Streetview")
    evaluateStreetview(Predictor)
  else:
    print("Unknown dataset type '%s'", Predictor.DATASET_TYPE)
    exit(1)
    
