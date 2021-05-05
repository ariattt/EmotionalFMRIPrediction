import re
import os
import glob
from os import listdir
import csv
import collections
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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

if len(csvs) != 5:
    terminate("should have 5 csv files")

# # simple interpolation strategy: duplicate every other input
# def interpolate_stimulus(stimulus):
#     if len(stimulus) != 120:
#         terminate("input should have of size 120 but has size " + str(len(input)))
#     longer = []
#     for i, img in enumerate(stimulus):
#         longer.append(img)
#         if i%2 == 1:
#             longer.append(img)
#     assert len(longer) == 180
#     return longer

# # find the names of the input images (as strings, e.g. "036a")
# trial_stimuli = []
# for trialfile in csvs:
#     stimulus = []
#     with open(trialfile) as csvfile:
#         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#         for row in reader:
#             # skip header
#             if row[0] == "stimulus" or row[0] == 'x': continue

#             cur = row[0] if row[0].startswith("POFA/fMRI_POFA/") else row[2]
#             match = re.search("\d{3}[a-c]", cur)
#             img_name = match.group() if match else "" #empty string as rest
#             stimulus.append(img_name)
#     stimulus = interpolate_stimulus(stimulus)
#     trial_stimuli.append(stimulus)

# # find the corresponding labels of the input images
# emo_classes = {"happy": [1, 7, 14, 22, 29, 34, 35, 42, 48, 57, 66, 73, 74,
#                             84, 85, 93, 100, 101],
#                 "sad": [2, 8, 15, 23, 36, 43, 49, 58, 67, 75, 76, 77, 86,
#                         87, 94, 102, 103],
#                 "fear": [9, 16, 17, 24, 37, 50, 51, 59, 60, 68, 78, 79,
#                         88, 95, 104],
#                 "anger": [3, 10, 18, 25, 30, 38, 44, 52, 53, 61, 62, 69, 80,
#                             89, 96, 105, 106],
#                 "surprise": [4, 11, 19, 26, 31, 39, 45, 54, 63, 70, 81, 90,
#                             97, 107],
#                 "disgust": [5, 12, 20, 27, 32, 40, 46, 55, 64, 71, 82, 91,
#                             98, 108, 109],
#                 "neutral": [6, 13, 21, 28, 33, 41, 47, 56, 65, 72, 83, 92,
#                             99, 110],
#                 }

# reverse_map = {}
# for emo in emo_classes:
#     for n in emo_classes[emo]:
#         reverse_map[n] = emo

# trial_labels = []
# for stimulus in trial_stimuli:
#     labels = []
#     for img_name in stimulus:
#         if img_name == "":
#             labels.append("rest")
#         elif img_name[-1] != 'a':
#             labels.append("scrambled")
#         else:
#             num = int(img_name[:-1])
#             labels.append(reverse_map[num])
#     trial_labels.append(labels)

# # turn the labels into a one-hot encoding 
# # same number of labels, each with 6 possible classes
# one_hot_labels = np.zeros((*np.shape(trial_labels), 6))
# one_hot_enc = {
#     'neutral': 0, 'happy': 1, 'anger': 2, 'sad': 3, 'scrambled': 4, 'rest': 5
# }
# for i, labels in enumerate(trial_labels):
#     for j, label in enumerate(labels):
#         ind = one_hot_enc[label]
#         one_hot_labels[i, j, ind] = 1

# fmri = []
# for nii in niis:
#     if "rest" in nii:
#         continue
#     fmri.append(nib.load(nii).get_fdata())

# semantic_seq = one_hot_labels
# print("shape of semantic seq: ", semantic_seq.shape)
# # print(semantic_seq)

# n_delays = 5

# X, Y = [], []
# for run_x in semantic_seq:
#     # 5 runs from the dataset
#     # run_x, run_y = semantic_seq[i], fmri[i]
#     for i in range(180-n_delays):
#         X.append(run_x[i:i + n_delays].flatten())
# X = np.array(X)

# Y = np.array(Y)
# for run_y in fmri:
#     # print(run_y.shape)
#     if n_delays+175 >= 185: terminate("n_delay is too large")
#     if len(Y) == 0: Y = run_y[:,:,:,n_delays:n_delays + 175]
#     else: Y = np.concatenate((Y, run_y[:,:,:,n_delays:n_delays + 175]), axis=3)

# print(X.shape)
# # (875, 30)
# print(Y.shape)
# # (64, 64, 35, 875)
# # should be able to use built-in package from here

# def stat(result):
#     if result.shape != (64, 64, 35): terminate("result should be of size (64, 64, 35)")
#     avg = np.sum(result) / (64 * 64 * 35)
#     median = np.median(result.flatten())
#     res_min = np.min(result)
#     res_max = np.max(result)
#     print("min: ", res_min)
#     print("max: ", res_max)
#     print("average: ", avg)
#     print("median: ", median)

baseline_result = []
from sklearn.linear_model import LogisticRegression
for xx in range(64):
    for yy in range(64):
        for zz in range(35):
            correct = sum([1 if response-0 < 0.1 else 0 for response in Y[xx, yy, zz, :]])
            baseline_result.append(correct / 875)
baseline_result = np.array(baseline_result).reshape(64, 64, 35)
np.save('baseline.npy', baseline_result)

import seaborn as sns

def draw_result(result):
    if result.shape != (64, 64, 35): terminate("result should be of size (64, 64, 35)")
    try: os.makedirs(pwd + "/images")
    except: pass
    x_range = range(64)
    y_range = range(64)

    fig, axs = plt.subplots(6,6, sharey='all', sharex='all')
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for zz in range(35):
        display_cbar = True if zz == 17 else False
        g = sns.heatmap(result[:,:,zz], ax=axs[zz // 6][zz % 6], cbar=display_cbar, cbar_ax = cbar_ax if zz == 17 else None)
    plt.show()
draw_result(np.load('baseline.npy'))

# display(baseline_result)

            # try:
            #     clf = LogisticRegression(random_state=0).fit(X, Y[xx,yy,zz,:])
            #     print(clf.score(X, Y[xx,yy,zz,:]))
            # except ValueError as e: 
            #     print("voxel {xx} {yy} {zz}:")
            #     print(e)

