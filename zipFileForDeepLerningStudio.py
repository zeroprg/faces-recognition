import os
from os import listdir
from os.path import isfile, join
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to directory  with subfolders of images ")
args = vars(ap.parse_args())

folder  = args["path"]
subfolders = [f for f in os.scandir(folder) if f.is_dir() ]    
with open('train.csv', 'w') as the_file:
    the_file.write('name,rate\n')
i = 0
for subfolder in subfolders:
    print ("Folder: " + subfolder.path +"\n")    
    onlyfiles = [f for f in listdir(subfolder.path) if isfile(join(subfolder.path, f))]
    for file in onlyfiles:
        #print (file)
        with open('train.csv', 'a') as the_file:
            the_file.write("./"+ subfolder.name +"/" + file+"," +str(i)+ '\n')
    i+=1