# ref: http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
# ref: https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#batching
# ref: https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

import numpy as np
import tensorflow as tf

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))

    unique = list(set(labels))
    i2label = {i:x for (i,x) in enumerate(unique)}
    label2i = {x:i for (i,x) in enumerate(unique)}
    # convert string-represented labels to integer representation
    labels = [label2i[x] for x in labels]
    return filenames, labels, i2label, label2i  
    

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


print("  ----   input_pipeline.py is imported -----")

LABELS_FILE = './smallYearbookF.labels.txt'
BATCH_SIZE = 100
SHAPE= (171*186*3) # [height, width] This cannot read from file and needs to be provided
num_epochs = None # If set to a number, will generate OutOfRange error after $num_epochs repetation
# Reads pfathes of images together with their labels
images, labels, i2label, label2i = read_labeled_image_list(LABELS_FILE)
LABEL_CNT = len(i2label)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)

one_image, one_label = read_images_from_disk(input_queue)
one_image = tf.image.convert_image_dtype(one_image, tf.float32)
one_image = tf.reshape(one_image, [-1]) # [-1] means flatten the tensor

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
# one_image = preprocess_image(one_label)
# one_image = preprocess_label(one_label)

# Batching (input tensors backed by a queue; and then combine inputs into a batch)
image_batch, label_batch = tf.train.batch([one_image, one_label],
                                           batch_size=BATCH_SIZE,
                                           shapes=[SHAPE, ()])
                                           
                                           
if __name__ == "__main__":
    sess = tf.Session()

    # Required. 
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    img, label = sess.run([one_image, one_label])
    print(img)
    print(label)

