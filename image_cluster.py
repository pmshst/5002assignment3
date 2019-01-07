#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python version 3.6.5
# using pep8 pycodestyle
from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
import PIL.Image
import shutil
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import os
import re
import pickle
path_join = os.path.join
imagecluster_base_dir = 'cluster_output'


def get_model():
    '''
    get VGG16 with the output layer set to
    the second-to-last fully connected layer 'fc2' of shape 4096
    :return:
    '''
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    return model


def get_features_one_image_file(filename, model, size):
    '''
    Load image from filenames, resize to size and run through `model`
    (keras.models.Model).
    :param filename:
    :param model:
    :param size:
    :return: 1d array
    '''
    img = PIL.Image.open(filename).resize(size, 3)
    array_3d = image.img_to_array(img)
    # (224, 224, 1) -> (224, 224, 3)
    # Simple hack to convert a grayscale image to
    # fake RGB by replication of
    # the image data to all 3 channels.
    # Deep learning models may have
    # learned color-specific filters, but the
    # assumption is that structural image features (edges etc)
    # contibute more to
    # the image representation than color,
    # such that this hack makes it possible
    # to process gray-scale images with nets trained on
    # color images (like VGG16).
    if array_3d.shape[2] == 1:
        array_3d = array_3d.repeat(3, axis=2)
    # (1, 224, 224, 3)
    array_4d = np.expand_dims(array_3d, axis=0)
    # (1, 224, 224, 3)
    input = preprocess_input(array_4d)
    return model.predict(input)[0, :]


def get_features_from_all_images(files, model, size=(224, 224)):
    '''
    :param files: image filenames
    :param model:
    :param size:
    :return:
    features : dict
        {filename1: array([...]),
         filename2: array([...]),
         ...
         }
    '''
    return dict((file_name, get_features_one_image_file(file_name, model, size)) for file_name in files)


def cluster_dendrogram(features_dict, sim=0.6, method='average', metric='euclidean'):
    '''
    Hierarchical clustering of images based on features get from VGG16 model(fc2)
    :param features_dict:
    :param sim: float 0..1
        similarity index
    :param method: see scipy.hierarchy.linkage(), all except 'centroid' produce
        pretty much the same result
    :param metric: see scipy.hierarchy.linkage(), make sure to use 'euclidean' in
        case of method='centroid', 'median' or 'ward'
    :return:
    clusters : nested list
        [[filename1, filename5],                    # cluster 1
         [filename23],                              # cluster 2
         [filename48, filename2, filename42, ...],  # cluster 3
         ...
         ]
    '''
    assert 0 <= sim <= 1, "sim not 0..1"
    # Pairwise distances between observations in n-dimensional space.
    distances_image = distance.pdist(
        np.array(list(features_dict.values())),
        metric)
    imae_files_path_list = list(features_dict.keys())
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram)
    Z = hierarchy.linkage(distances_image, method=method, metric=metric)
    # cut dendrogram and extract clusters
    # Form flat clusters from the hierarchical clustering defined by
    # the given linkage matrix.
    T = hierarchy.fcluster(
        Z,
        t=distances_image.max()*(1.0-sim),
        criterion='distance')
    cluster_dict = dict((index, []) for index in np.unique(T))
    for img_index, cluster_index in enumerate(T):
        cluster_dict[cluster_index].append(imae_files_path_list[img_index])
    return list(cluster_dict.values())


def save_results(image_clusters, clusters_save_dir):
    '''
    save cluster results to prediction.csv
     group all image_clusters by size
     (cluster = list_of_files) of equal size together
     {number_of_files1: [[list_of_files], [list_of_files],...],
      number_of_files2: [[list_of_files],...],
     }
    :param image_clusters: image clusters with image file paths
    :param clusters_save_dir:
    :return:
    '''
    clusters_dic = {}
    clusters_dic_to_csv = {}
    i = 0
    for cluster in image_clusters:
        cluster_size = len(cluster)
        # group cluters by size
        if not (cluster_size in clusters_dic.keys()):
            clusters_dic[cluster_size] = [cluster]
        else:
            clusters_dic[cluster_size].append(cluster)
    max_size = np.max(list(clusters_dic.keys()))
    for cluster in image_clusters:
        i += 1
        tmp_list = []
        for j in cluster:
            # get image id like 05001
            tmp_list.append(j[-9:-4])
        clusters_dic_to_csv['Cluster ' + str(i)] = tmp_list
        if len(cluster) < max_size:
            t = len(cluster)
            while t < max_size:
                clusters_dic_to_csv['Cluster ' + str(i)].append('')
                t += 1
    df = pd.DataFrame(clusters_dic_to_csv)
    df.to_csv('prediction.csv', index=False)
    print("cluster dir: {}".format(clusters_save_dir))
    print("cluster size : number of clusters with the same size")
    if os.path.exists(clusters_save_dir):
        shutil.rmtree(clusters_save_dir)
    cluster_total_num = len(image_clusters)
    for cluster_size in np.sort(list(clusters_dic.keys())):
        cluster_list = clusters_dic[cluster_size]
        print("{} : {}".format(cluster_size, len(cluster_list)))
        for cluster_id, filepath_list in enumerate(cluster_list):
            cluster_dir = path_join(
                     clusters_save_dir,
                     'cluster_with_{}'.format(cluster_size),
                     'cluster_{}'.format(cluster_id))
            for filepath in filepath_list:
                link = path_join(cluster_dir, os.path.basename(filepath))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                # link to source image file
                os.symlink(os.path.abspath(filepath), link)
    print('cluster_total_num', cluster_total_num)


def load_pkl(file_name):
    '''
    load VGG16features.pkl
    :param file_name:
    :return:
    '''
    with open(file_name, 'rb') as pkl:
        ret = pickle.load(pkl)
    return ret


def dump_pkl(features, file_name):
    '''
    dump VGG16features.pkl
    :param features:
    :param file_name:
    :return:
    '''
    with open(file_name, 'wb') as pkl:
        pickle.dump(features, pkl)


def load_image_files(file_dir, ext='jpg|jpeg|bmp|png'):
    '''
    load image files
    :param file_dir:
    :param ext:
    :return:
    '''
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(file_dir, base) for base in os.listdir(file_dir) if rex.match(base)]


def main(image_dir, sim=0.6):
    '''
    :param image_dir: path to directory with images
    :param sim: float (0..1)
    :return:
    '''
    vgg16_features_pkl = path_join(
        image_dir,
        imagecluster_base_dir,
        'VGG16features.pkl')
    # fist time save generate VGG16features and dump them
    if not os.path.exists(vgg16_features_pkl):
        os.makedirs(os.path.dirname(vgg16_features_pkl), exist_ok=True)
        print("no VGG16features.pkl found".format(vgg16_features_pkl))
        files = load_image_files(image_dir)
        # get vgg16 model
        model = get_model()
        print("generating VGG16features.pkl ".format(vgg16_features_pkl))
        features = get_features_from_all_images(files, model, size=(224, 224))
        dump_pkl(features, vgg16_features_pkl)
    else:
        print("loading VGG16features.pkl".format(vgg16_features_pkl))
        features = load_pkl(vgg16_features_pkl)
        print("clustering ..\n sim=", sim)
    save_results(
        cluster_dendrogram(features, sim),
        path_join(image_dir, imagecluster_base_dir, 'clusters'))


# for i in np.arange(0.51, 0.6, 0.01):
main('/Users/zhaocai/Downloads/courses/MSBD5002-Knowledge_Discovery_and_Data_Mining/assignment_3/images', 0.6)


