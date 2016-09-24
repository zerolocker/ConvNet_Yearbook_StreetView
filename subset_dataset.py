import os
import numpy as np
import shutil

YEARBOOK_DATA = './yearbook'
SUBSET_OUTPUT_PATH = './smallYearbook/'
SUBSET_OUTPUT_PATH_TRAIN = SUBSET_OUTPUT_PATH + 'train/'
SUBSET_OUTPUT_PATH_VAL = SUBSET_OUTPUT_PATH + 'val/'

def ls_relpath_r(path): # recursively list files
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]
    
    
files = ls_relpath_r(YEARBOOK_DATA)
files = filter(lambda x: x.endswith('.png'), files)

labels = [os.path.basename(x).split('_')[0] for x in files]

# group the filenames by labels
samples_for_each_class = {label: [] for label in labels}
for i in range(len(files)):
    samples_for_each_class[labels[i]].append(files[i])
    

if not os.path.exists(SUBSET_OUTPUT_PATH_TRAIN):
    os.mkdir(SUBSET_OUTPUT_PATH)
if not os.path.exists(SUBSET_OUTPUT_PATH_TRAIN):
    os.mkdir(SUBSET_OUTPUT_PATH_TRAIN)
if not os.path.exists(SUBSET_OUTPUT_PATH_VAL):
    os.mkdir(SUBSET_OUTPUT_PATH_VAL)
    
ftrain = open(SUBSET_OUTPUT_PATH + 'label.train.txt', 'w')
fval = open(SUBSET_OUTPUT_PATH + 'label.val.txt', 'w')
for (label,samples) in samples_for_each_class.items():
    # randomly pick 10 samples to train, 2 samples to val
    TRAIN_CNT = 10; VAL_CNT = 2
    print("sample size for label %s is %d" %(label, len(samples)))
    if len(samples)  < TRAIN_CNT + VAL_CNT: # not enough sample for training data
        VAL_CNT = min(VAL_CNT, len(samples))
        TRAIN_CNT = min(TRAIN_CNT, len(samples) - VAL_CNT)
        assert(VAL_CNT > 0 and TRAIN_CNT >=0)
        
    rndidx = np.random.choice(len(samples), TRAIN_CNT+VAL_CNT, replace=False)
    samples = np.array(samples) # convert to ndarray
    subset = samples[rndidx]
    for filename in subset[:TRAIN_CNT]:
        shutil.copy(filename, SUBSET_OUTPUT_PATH_TRAIN + os.path.basename(filename))
        ftrain.write('%s %s' % (filename, label))
    for filename in subset[TRAIN_CNT : TRAIN_CNT+VAL_CNT]:
        shutil.copy(filename, SUBSET_OUTPUT_PATH_VAL + os.path.basename(filename))
        fval.write('%s %s' % (filename, label))
print("label count is :%d" % (len(samples_for_each_class)))

ftrain.close()
fval.close()

