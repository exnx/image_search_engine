
# coding: utf-8

# # CS 5785 - Final
# ## Applied Machine Learning
# 
# Arpit Sheth *(as3668)*
# 
# Eric Nguyen *(en274)*
# 
# Samantha Yip *(sty5)*

# In[1]:

import numpy as np
import pandas as pd
from PIL import Image
import random


# # Constants and Variables

# In[2]:

TRAIN = 0
TEST = 1
N_TRAIN = 10000
N_TEST = 2000

# These become Pandas DataFrames
descriptions_train = []
descriptions_test = []
features_train = []
features_test = []
ifeatures_train = []
ifeatures_test = []
tags_train = []
tags_test = []


# ---
# # Input Functions 

# # Descriptions

# In[3]:

def read_description(i, dataset):
    """Returns the i-th description from the training or testing dataset
    as a list of length 5. Filters out empty lines and any sentences after
    the first five sentences."""
    filepath = 'descriptions_train/' if dataset == TRAIN else 'descriptions_test/'
    filepath += str(i) + ".txt"
    with open(filepath) as f:
        lines = f.read().splitlines()
        lines = list(filter(None, lines))
    return lines[:5]


def get_descriptions(n, dataset):
    """Returns a pandas dataframe of all the descriptions in a the training or testing dataset."""
    descriptions = []
    for i in range(n):
        descriptions.append([read_description(i, dataset)])

    descriptions = pd.DataFrame(descriptions)
    descriptions = pd.DataFrame(descriptions[0].values.tolist(), columns=["s1", "s2", "s3", "s4", "s5"])
    return descriptions 


# In[4]:

descriptions_train = get_descriptions(N_TRAIN, TRAIN)
descriptions_test = get_descriptions(N_TEST, TEST)


# # Features and Intermediate Features

# In[5]:

def get_features(intermediate, dataset):
    """Returns the features for training or testing dataset"""
    heading = ["id"]
    
    if intermediate:
        filepath = "features_train/features_resnet1000intermediate_train.csv" if dataset == TRAIN else "features_test/features_resnet1000intermediate_test.csv"
        heading.extend(list(range(2048)))
    else:
        filepath = "features_train/features_resnet1000_train.csv" if dataset == TRAIN else "features_test/features_resnet1000_test.csv"
        heading.extend(list(range(1000)))

    features = pd.read_csv(filepath, names=heading)  
    
    ids = []
    for index, row in features.iterrows():
        fn = row["id"]
        start = fn.find("/") + 1
        end = -4
        i = int(fn[start:end])
        ids.append(i)
    
    features = features.assign(id = ids)
    features = features.sort_values("id")
    features = features.reset_index(drop=True)
    features = features.drop('id', axis=1)
    return features


# In[6]:

features_train = get_features(False, TRAIN)
features_test = get_features(False, TEST)
ifeatures_train = get_features(True, TRAIN)
ifeatures_test = get_features(True, TEST)


# # Images

# In[7]:

def get_image(i, dataset):
    """Returns an image to be displayed."""
    filepath = 'images_train/' if dataset == TRAIN else 'images_test/'
    filepath += str(i) + ".jpg"
    img = Image.open(filepath)
    return img


def get_imagedata(i, dataset):
    """Returns an image's pixel values as a numpy array."""
    return np.array(get_image(i, dataset))


# # Tags

# In[8]:

def read_tags(i, dataset):
    """Returns the i-th image tags from the training or testing dataset
    as a list of tuples with format (supercategory, category)."""
    filepath = 'tags_train/' if dataset == TRAIN else 'tags_test/'
    filepath += str(i) + ".txt"
    with open(filepath) as f:
        lines = f.read().splitlines()
        lines = list(filter(None, lines))
    imgtags = []
    for tag in lines:
        imgtags.append(tuple(tag.split(':')))
    return imgtags


def get_tags(n, dataset):
    """Returns a pandas dataframe of all the image tags in a the training or testing dataset."""
    tags = []
    for i in range(n):
        tags.append([read_tags(i, dataset)])
    tags = pd.DataFrame(tags, columns=["tags"])
    return tags 


# ---
# # Output

# In[9]:

def save_output(output, filename):
    """Given an output Pandas DataFrame, save it according to the required format.
    The output Pandas DataFrame should have format of N_TEST rows by 21 columns
    where the first column is the integer corresponding to the description query
    and the next 20 columns are the ranked predictions of images."""
    output_formatted = []
    for index, row in output.iterrows():
        description = str(row[0]) + ".txt"
        images = row[1:].tolist()
        images = [str(img) + ".jpg" for img in images]
        images = ' '.join(images)
        output_formatted.append([description, images])
    output_formatted = pd.DataFrame(output_formatted, columns=["Descritpion_ID", "Top_20_Image_IDs"])
    output_formatted.to_csv(filename, index=False)
    return output_formatted  


# In[10]:

def save_output_random(filename):
    output = []
    for i in range(N_TEST):
        prediction = random.sample(range(0, N_TEST), 20)
        row = [i]
        row.extend(prediction)
        output.append(row)
    output = pd.DataFrame(output)
    return save_output(output, filename)
