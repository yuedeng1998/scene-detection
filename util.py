import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

import pickle
def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    keypoints = np.zeros((n_image * n_each, 2))
    descriptors = np.zeros((n_image * n_each, 128))
    
    #print(np.shape(descriptors))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)
        sift_keypoints = features[:, :2]
        sift_descriptors = features[:, 2:]
        # TODO: Randomly sample n_each descriptors from sift_descriptor and store them into descriptors
        sample_list = random.sample(list(sift_descriptors), n_each)
        descriptors[i*n_each:(i+1)*n_each] = sample_list;


    # TODO: pefrom k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters= vocab_size, random_state = 0).fit(descriptors)
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each feature to the closest cluster center
        # Again, each feature consists of the (x, y) location and the 128-dimensional sift descriptor
        # You can access the sift descriptors part by features[:, 2:]

        #all the descriptors for one image
        sift_descriptors = features[:,2:]
        #the number of descriptors for this image
            #find the index of the closest centroid
        # TODO: Build a histogram normalized by the number of descriptors
        index = kmeans.predict(sift_descriptors)
            #add one to that clauser for this image
        for m in index:
            image_feats[i, m] += 1
        des_num = len(sift_descriptors)
        #normalizing the histogram
        image_feats[i] = image_feats[i]/des_num    
    return image_feats

def plo_class_histg(image_feats, image_labels, kmeans):
    #import pdb; pdb.set_trace()
    #get v_size
    vocab_size = kmeans.cluster_centers_.shape[0]
    #store the histogram for each cluaster
    histogram = np.zeros((15, vocab_size))
    #counting how many images within each cluaster
    counting = np.zeros((15, 1))
    #for each image counting and adding these histgrams in each imgine
    for i in range(len(image_feats)):
        label = image_labels[i]
        histogram[int(label)] += image_feats[i]
        counting[int(label)] += 1
    #normalize the hisgrams
    for j, his in enumerate(histogram):
        his = his/counting[j]
        index = np.arange(300)
        plt.figure()
        plt.bar(index, his)
    #show the histragms
    plt.show()

def load(ds_path):
    """ Load from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    labels: class labels corresponding to each image
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]
    return image_paths, labels


if __name__ == "__main__":
    paths, labels = load("sift/train")

    kmeans = build_vocabulary(paths, 3)
    feat = get_bags_of_sifts(paths, kmeans)
    
    #build_vocabulary(paths, 10)
