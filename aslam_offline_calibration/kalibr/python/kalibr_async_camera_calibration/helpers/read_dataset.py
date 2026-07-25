'''
Reads dataset, topic from bagfile.

Also gives Number of images in the bag.
'''

import kalibr_common as kc

#read image topics from the dataset
def initBagDataset(bagfile, topic, from_to, freq):
    print("\tDataset:   {0}".format(bagfile))
    print("\tTopic:     {0}".format(topic))
    reader = kc.BagImageDatasetReader(bagfile, topic, bag_from_to=from_to, bag_freq=freq)
    print("\tNumber of images in the bag: {0}".format(reader.numImages()))
    return reader