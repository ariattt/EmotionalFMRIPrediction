import re
import os
import glob
from os import listdir
import csv
import collections
import numpy as np
import nibabel as nib
from sklearn import preprocessing
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

# simple interpolation strategy: duplicate every other input
def interpolate_stimulus(stimulus):
    if len(stimulus) != 120:
        terminate("input should have of size 120 but has size " + str(len(input)))
    longer = []
    for i, img in enumerate(stimulus):
        longer.append(img)
        if i%2 == 1:
            longer.append(img)
    assert len(longer) == 180
    return longer

# find the names of the input images (as strings, e.g. "036a")
trial_stimuli = []
for trialfile in csvs:
    stimulus = []
    with open(trialfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            # skip header
            if row[0] == "stimulus" or row[0] == 'x': continue

            cur = row[0] if row[0].startswith("POFA/fMRI_POFA/") else row[2]
            match = re.search("\d{3}[a-c]", cur)
            img_name = match.group() if match else "" #empty string as rest
            stimulus.append(img_name)
    stimulus = interpolate_stimulus(stimulus)
    trial_stimuli.append(stimulus)

# find the corresponding labels of the input images
emo_classes = {"happy": [1, 7, 14, 22, 29, 34, 35, 42, 48, 57, 66, 73, 74,
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

reverse_map = {}
for emo in emo_classes:
    for n in emo_classes[emo]:
        reverse_map[n] = emo

trial_labels = []
for stimulus in trial_stimuli:
    labels = []
    for img_name in stimulus:
        if img_name == "":
            labels.append("rest")
        elif img_name[-1] != 'a':
            labels.append("scrambled")
        else:
            num = int(img_name[:-1])
            labels.append(reverse_map[num])
    trial_labels.append(labels)

# turn the labels into a one-hot encoding 
# same number of labels, each with 6 possible classes
one_hot_labels = np.zeros((*np.shape(trial_labels), 6))
one_hot_enc = {
    'neutral': 0, 'happy': 1, 'anger': 2, 'sad': 3, 'scrambled': 4, 'rest': 5
}
for i, labels in enumerate(trial_labels):
    for j, label in enumerate(labels):
        ind = one_hot_enc[label]
        one_hot_labels[i, j, ind] = 1

fmri = []
for nii in niis:
    if "rest" in nii:
        continue
    fmri.append(nib.load(nii).get_fdata())

semantic_seq = one_hot_labels
print("shape of semantic seq: ", semantic_seq.shape)
# print(semantic_seq)

n_delays = 5

X, Y = [], []
for run_x in semantic_seq:
    # 5 runs from the dataset
    # run_x, run_y = semantic_seq[i], fmri[i]
    for i in range(180-n_delays):
        X.append(run_x[i:i + n_delays].flatten())
X = np.array(X)

Y = np.array(Y)
for run_y in fmri:
    print(run_y.shape)
    if n_delays+175 >= 185: terminate("n_delay is too large")
    if len(Y) == 0: Y = run_y[:,:,:,n_delays:n_delays + 175]
    else: Y = np.concatenate((Y, run_y[:,:,:,n_delays:n_delays + 175]), axis=3)

print(X.shape)
# (875, 30)
print(Y.shape)
# (64, 64, 35, 875)


def ols(x, y):
    beta_ols = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return beta_ols

# n_train=700
# n_test=175

# X_train=X[:n_train,:]
# Y_train=Y[:,:,:,:n_train]
# X_test=X[n_train:,:]
# Y_test=Y[:,:,:,n_train:]

# beta_ols=np.zeros((64, 64, 35, 30))
# val_ses_ols=np.zeros((64, 64, 35))
# cnt=0
# for xx in range(64):
#     for yy in range(64):
#         for zz in range(35):
#             Y_curr_train = Y_train[xx,yy,zz,:]
#             Y_curr_test = Y_test[xx,yy,zz,:]
#             beta_ols[xx,yy,zz,:]=ols(X_train, Y_curr_train)
#             Y_test_hat = X_test.dot(beta_ols[xx,yy,zz,:])
#             val_ses_ols[xx,yy,zz]=np.sum((Y_curr_test - Y_test_hat) ** 2)/n_test
#             cnt += 1
#             print(cnt)



def ridge(x, y, lam):
    n_features = x.shape[1]
    beta_ridge = np.linalg.inv(x.T.dot(x) + lam * np.eye(n_features)).dot(x.T).dot(y)
    return beta_ridge

def get_best_lambda_val(x,y):
    k=4
    n_trn=x.shape[0]
    n_per_fold = n_trn/k

    lambdas = np.logspace(-3, 5, 10)
    val_mses = np.zeros((64, 64, 35, k, len(lambdas)))

    for fold in range(k):
        print(f'fold:{fold}')
        x_trn = np.delete(x, np.s_[fold * int(n_per_fold):(fold + 1) * int(n_per_fold)], 0)
        x_val = x[fold * int(n_per_fold):(fold + 1) * int(n_per_fold), :]
        
        for xx in range(64):
            for yy in range(64):
                for zz in range(35):
                    y_trn = np.delete(y[xx,yy,zz,:], np.s_[fold * int(n_per_fold):(fold + 1) * int(n_per_fold)], 0)
                    y_val = y[xx,yy,zz,:][fold * int(n_per_fold):(fold + 1) * int(n_per_fold)]
                    print(f'voxel position:{xx},{yy},{zz}')
                    for ii in range(len(lambdas)):
                        y_val_hat = x_val.dot(ridge(x_trn, y_trn, lambdas[ii]))
                        val_mses[xx,yy,zz,fold,ii] = np.average((y_val - y_val_hat) ** 2)
    
    
    lambda_mse = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        lambda_mse[i] = np.sum(val_mses[:,:,:,:,i],axis=(0,1,2,3)) / k
        
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_xscale('log')
    plt.plot(lambdas, lambda_mse, marker='o', linestyle='dashed')
    best_lambda = lambdas[np.argmin(lambda_mse)]
    return bset_lambda
        

n_train=700
n_test=175

X_train=X[:n_train,:]
Y_train=Y[:,:,:,:n_train]
X_test=X[n_train:,:]
Y_test=Y[:,:,:,n_train:]

# best_lambda_val = get_best_lambda_val(X_train, Y_train)
best_lambda_val = 0.464158883
print(f'best estimated lambda val:{best_lambda_val}')
beta_ridge=np.zeros((64, 64, 35, 30))
val_ses=np.zeros((64, 64, 35))
Y_test_hat=np.zeros((64, 64, 35, 175))
cnt=0
for xx in range(64):
    for yy in range(64):
        for zz in range(35):
            Y_curr_train = Y_train[xx,yy,zz,:]
            Y_curr_test = Y_test[xx,yy,zz,:]
            beta_ridge[xx,yy,zz,:]=ridge(X_train, Y_curr_train, best_lambda_val)
            Y_test_hat[xx,yy,zz,:] = X_test.dot(beta_ridge[xx,yy,zz,:])
            val_ses[xx,yy,zz]=np.sum((Y_curr_test - Y_test_hat) ** 2)/n_test
            cnt += 1
            print(cnt)

np.save('mse_ridge_regression.npy', val_ses)

# with open('test_hat.txt','w') as f:
#     for xx in range(64):
#         for yy in range(64):
#             for zz in range(35):
#                 f.write(str(Y_test_hat[xx,yy,zz,:])+'\n')