# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:29:31 2018

@author: Hendry
"""

import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = 'dataset/FaceMask/'

# Percentage of images to be used for the test set
percentage_test = 30;

# Create and/or truncate train.txt and test.txt
file_train = open('trainFaceMask.txt', 'w')  
file_test = open('testFaceMask.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1