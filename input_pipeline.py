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

    labels = [label2i(x) for x in labels]
    return filenames, labels
    

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
    
    
def create_input_pipeline(LABEL_FILE, BATCH_SIZE, num_epochs, produceVGGInput): 
    """
        num_epochs: a number, or None. If it's a number, the queue will generate OutOfRange error after $num_epochs repetations
    """
    images, labels = read_labeled_image_list(LABEL_FILE)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    one_image, one_label = read_images_from_disk(input_queue)
    one_image = tf.image.convert_image_dtype(one_image, tf.float32)
    if produceVGGInput:  # resize it to VGG input
        one_image = tf.image.resize_images(one_image, 224, 224)
    else: 
        one_image = tf.reshape(one_image, [-1]) # [-1] means flatten the tensor

    # Optional Preprocessing or Data Augmentation
    # if you have time, look at https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#preprocessing

    # Batching (input tensors backed by a queue; and then combine inputs into a batch)
    image_batch, label_batch = tf.train.batch([one_image, one_label],
                                               batch_size=BATCH_SIZE,
                                               shapes=[SHAPE, ()], capacity=3*BATCH_SIZE)
    return image_batch, label_batch, len(images)
    

def printdebug(str):
    print('  ----   DEBUG: '+str)

print("  ----   input_pipeline.py is imported -----")

# dataset-specific definitions
LABEL_CNT = 109;
LABELS_FILE_TRAIN = './smallYearbook/label.train.txt'
# LABELS_FILE_TRAIN = './smallYearbookF.labels.txt'
LABELS_FILE_VAL = './smallYearbook/label.val.txt'
BATCH_SIZE = 100
SHAPE= (171*186*3) # [height, width] This cannot read from file and needs to be provided
i2label = lambda i: i+1905;
label2i = lambda i: i-1905;


if __name__ == "__main__":
    sess = tf.Session()
    
    # create input pipelines for training set and validation set
    train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, BATCH_SIZE, num_epochs=None, produceVGGInput=False)
    val_image_batch, val_label_batch, VAL_SIZE = create_input_pipeline(LABELS_FILE_VAL, BATCH_SIZE, num_epochs=None, produceVGGInput=False)
    printdebug("TRAIN_SIZE: %d VAL_SIZE: %d BATCH_SIZE: %d " % (TRAIN_SIZE, VAL_SIZE, BATCH_SIZE))

    # Required. 
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    print ("  ----   Running unit tests for training set -----")
    imgs, labels = sess.run([train_image_batch, train_label_batch])
    assert(imgs.shape[0] == BATCH_SIZE)
    assert(imgs[0].reshape(-1)[0] != imgs[1].reshape(-1)[0])
    print(imgs[0])
    print(labels[0])
    imgs2, labels2 = sess.run([train_image_batch, train_label_batch])
    assert(imgs2[0].reshape(-1)[0] != imgs[0].reshape(-1)[0]) # test if the batches change between two calls to sess.run()

    print ("  ----   Running unit tests for validation set -----")
    for i in range(VAL_SIZE / BATCH_SIZE): 
        imgs, labels = sess.run([val_image_batch, val_label_batch])


