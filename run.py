import os
import glob
from os import listdir
import csv
import collections
import numpy as np
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
            # skip header
            if row[0] == "stimulus" or row[0] == 'x': continue

            cur = row[0] if row[0].startswith("POFA/fMRI_POFA/") else row[2]
            if not cur.startswith("POFA/fMRI_POFA"): 
                order.append(0) # 0 as rest
                continue

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
                    "rest": [0]
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

# print(fmri[0].shape)
def avg(l1, l2): return [(l1[i]+l2[i]) / 2 for i in range(len(l1))]

def inter_120_180(input):
    if len(input) != 120: terminate("input should have of size 120 but has size " + str(len(input)))
    res = []
    for i in range(119):
        res.append(input[i])
        if i % 2 == 0: 
            res.append(avg(input[i], input[i+1]))
    last = [ input[119][i]*2 - input[118][i] for i in range(len(input[119]))]
    res.append(last)
    return res


semantic_seq = []
map = {'neutral': [1,0,0,0,0],
        'happy': [0,1,0,0,0],
        'anger': [0,0,1,0,0],
        'sad': [0,0,0,1,0],
        'rest': [0,0,0,0,1]}

for o in orders:
    cur = []
    for mood in o:
        cur.append(map[mood])
    cur = inter_120_180(cur)
    np_cur = np.array(cur)
    semantic_seq.append(np_cur)


train_x, train_y, test_x, test_y = semantic_seq[:4], fmri[:4], semantic_seq[4:], fmri[4:]
n_delays = 5

X, Y = [], []
for run_x in semantic_seq:
    # 5 runs from the dataset
    # run_x, run_y = semantic_seq[i], fmri[i]
    for i in range(180-n_delays):
        X.append(run_x[i:i + n_delays].flatten())
X = np.array(X)
print(X.shape)

Y = np.array(Y)
for run_y in fmri:
    # print(run_y.shape)
    if n_delays+175 >= 185: terminate("n_delay is too large")
    if len(Y) == 0: Y = run_y[:,:,:,n_delays:n_delays + 175]
    else: Y = np.concatenate((Y, run_y[:,:,:,n_delays:n_delays + 175]), axis=3)

print(Y[0].shape)

'''
# writing everything by hand

beta = np.zeros((64, 64, 35, 5, 5))
def predict(stim, beta):
    if stim.shape != (180, 5): terminate("stimulate should have of shape (180, 5)")
    if beta.shape != (5, 5): terminate("beta should be of shape (5, 5")
    pred = []
    for i in range(0, 175):
        res = np.sum(np.multiply(stim[i:i+5], beta))
        pred.append(res)
    return np.array(pred)

alpha = 1.0
def compute_error(xx, yy, zz, intensity, target, bb):
    if len(intensity) != 175: terminate("intensity should of size 175")
    if len(target) != 185: terminate("fmri should have of len 185")
    err = 0
    for i in range(5, 180):
        diff = intensity[i-5] - target[i]
        err += diff ** 2
    # regularization
    err += alpha * np.sum(bb ** 2)
    return err

predicted = []
num_iter = 100
thres = 1
errors = np.zeros((64, 64, 35))
for xx in range(64):
    for yy in range(64):
        for zz in range(35):
            errors = []
            for training_epoch in range(num_iter):
                err1 = err2 = 0
                eps = 0.0000000000001
                for i in range(len(train_x)):
                    stim = train_x[i]
                    bb = beta[xx][yy][zz]
                    intensity = predict(stim, bb)
                    predicted.append(intensity)
                    err1 += compute_error(xx, yy, zz, intensity, fmri[i][xx][yy][zz], bb)
                    intensity = predict(stim, bb+eps)
                    err2 += compute_error(xx, yy, zz, intensity, fmri[i][xx][yy][zz], bb+eps)
                # training finished for that cell
                if err1 < thres: break

                derivative = (err2 - err1) / eps
                step = 0.0000001
                bb -= step * derivative
                errors.append(err1)
            print(errors)
            # terminate("stop it")

# predicted = np.array(predicted).reshape(4, 64, 64, 35, 175)
# errors = np.divide(errors, 4)

# train_y = 

                
    # predict(train_x)

'''