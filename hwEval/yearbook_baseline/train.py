# This is an ugly hack to use util.py in the src folder. DO NOT COPY THIS, BUT
# RATHER PUT ALL YOUR CODE IN src FOLDER!!
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# END OF UGLY HACK

from util import *

THIS_DIRECTORY = path.dirname(path.abspath(__file__))

# Load all the training files 
yb = listYearbook()

# Get the labels
import numpy as np
years = np.array([label(y) for y in yb])

# Train
med = np.median(years, axis=0)

# Save the model
open(path.join(THIS_DIRECTORY,'model.txt'),'w').write('%d\n'%med)
