import os
import glob
from os import listdir
import csv
import collections
import numpy as numpy
import nibabel as nib

def find_csv(dir):
    return [dir + "/" + file for file in listdir(dir) if file.endswith(".csv")]

def find_nii(dir):
    return [dir + "/" + file for file in listdir(dir) if file.endswith(".nii")]

def terminate(msg):
    print(msg)
    exit()

pwd = os.getcwd()
csvs = find_csv(pwd + "/orders")
niis = find_nii(pwd + "/fmri")

if len(csvs) != 5:  terminate("should have 5 csv files")

orders = []
for file in csvs:
    order = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            cur = row[0] if row[0].startswith("POFA/fMRI_POFA/") else row[2]
            if not cur.startswith("POFA/fMRI_POFA"): continue

            s_index = cur.find('/', 8)
            tar = []
            for c in cur[s_index+1:]:
                if c >= '0' and c <= '9':
                    tar.append(c)
                else: break
            order.append(int("".join(tar)))
    orders.append(order)

def get_map():
    emotion_classes = {"happy": [1, 7, 14, 22, 29, 34, 35, 42, 48, 57, 66, 73, 74,
                             84, 85, 93, 100, 101],
                   "sad": [2, 8, 15, 23, 36, 43, 49, 58, 67, 75, 76, 77, 86,
                           87, 94, 102, 103],
                   "fear": [9, 16, 17, 24, 37, 50, 51, 59, 60, 68, 78, 79,
                            88, 95, 104],
                   "anger": [3, 10, 18, 25, 30, 38, 44, 52, 53, 61, 62, 69, 80,
                             89, 96, 105, 106],
                   "surprise": [4, 11, 19, 26, 31, 39, 45, 54, 63, 70, 81, 90,
                                97, 107],
                   "disgust": [5, 12, 20, 27, 32, 40, 46, 55, 64, 71, 82, 91,
                               98, 108, 109],
                   "neutral": [6, 13, 21, 28, 33, 41, 47, 56, 65, 72, 83, 92,
                               99, 110],
                   }
    map = {}
    for emo in emotion_classes:
        for n in emotion_classes[emo]:
            map[n] = emo
    return map

emotion_map = get_map()

for order in orders:
    for i in range(len(order)):
        order[i] = emotion_map[order[i]]

for o in orders:
    print(collections.Counter(o))

fmri = []
for nii in niis:
    if "rest" in nii: continue
    fmri.append(nib.load(nii).get_fdata())

print(fmri[0].shape)

