# This is an ugly hack to use util.py in the src folder. DO NOT COPY THIS, BUT
# RATHER PUT ALL YOUR CODE IN src FOLDER!!
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# END OF UGLY HACK

from util import *

THIS_DIRECTORY = path.dirname(path.abspath(__file__))

# Load all the training files 
sv = listStreetView()

# Get the labels
import numpy as np
coord = np.array([label(y) for y in sv])

# Compute the median.
# We do this in the projective space of the map instead of longitude/latitude,
# as France is almost flat and euclidean distances in the projective space are
# close enough to spherical distances.

xy = coordinateToXY(coord)
med = np.median(xy, axis=0, keepdims=True)
med_coord = np.squeeze(XYToCoordinate(med))

# Save the model
open(path.join(THIS_DIRECTORY,'model.txt'),'w').write('%f\n%f\n'%(med_coord[0], med_coord[1]))
