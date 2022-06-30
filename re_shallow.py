'''
-mode has noise and trainsz, -algo has KNN, RF, and SVM

Run with:
nohup python re_shallow.py -mode noise -algo KNN > knn.txt&
nohup python re_shallow.py -mode noise -algo RF > rf.txt&
nohup python re_shallow.py -mode noise -algo SVM > svm.txt&

nohup python re_shallow.py -mode trainsz -algo KNN > knn.txt&
nohup python re_shallow.py -mode trainsz -algo RF > rf.txt&
nohup python re_shallow.py -mode trainsz -algo SVM > svm.txt&
'''
import numpy as np
from numpy.random import seed 
seed(4) # fix seed for result reproducibility
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import os
from glob import glob
import json
import random
import pandas as pd
from data import AFM
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

from util import rf_, svm_, knn_

import time
import datetime

parser = argparse.ArgumentParser(description='AI-AFM')
parser.add_argument('-d','--data',type=str,default='3class_matching', choices=['3class_matching', '2class_s_vs_r', '2class_n_vs_r'], help="Choices of dataset (default='2class')")
parser.add_argument('-opt','--opt',type=str,default='sgd', choices=['sgd', 'adam', 'sam'], help="Choices of optimizer (default='sgd')")
parser.add_argument('-v','--verbosity', type=int, default=1, help='Verbosity. 0: No output, 1: Epoch-level output, 2: Batch-level output')
parser.add_argument('-g','--gpu', type=int, default=0, help='GPU id')
parser.add_argument('-ep','--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('-m','--model',type=str,default='toy', choices=['toy', 'toyL', 'toyS', 'toyS2', 'toyS3', 'toyXS', 'cerb', 'cerbL', 'cerbXL', 'convo1D', 'convo1DS', 'convo1DDrp', 'convo1DDrp2', 'convo1DS2', 'convo1D2', 'convo1D3'], help="Choices of models (default='toy')")
parser.add_argument('-mt','--model_type',type=str,default='siamese', choices=['siamese', 'triplet'], help="Model type (default='siamese')")
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('-bs','--batch_size', type=int, default=16, help='Training batch size (default = 16)')
parser.add_argument('-mod','--model_out_dim', type=int, default=10, help='Model output dim (default = 10)')
parser.add_argument('-log','--log_interval', type=int, default=10, help='Saving model at every specified interval (default = 10)')
parser.add_argument('-winsz','--winsz', type=int, default=5, help='AFM data processing window size (default = 5)')
parser.add_argument('-k','--top_k', type=int, default=1, help='Top-k majority for classification (default=1)')
parser.add_argument('-s','--seed', type=int, default=0, help='Random seed (default=0)')
parser.add_argument('-up','--unlabeled_percent', type=float, default=0.0, help='Percent of unlabeled data with range of [0.0,1.0) (default=0.0)')
parser.add_argument('-mgn','--margin', type=float, default=1.0, help='Loss function margin (default=1.0)')
parser.add_argument('-exp','--exp_num',type=str,default='0', help="Define experiment number")
parser.add_argument('-nt','--num_train_per_cls',type=int,default=-1, help="Number of training samples per class (default=-1, all data)")
parser.add_argument('-ds','--data_seed', type=int, default=0, help='Random seed for data sampling (default=0)')

parser.add_argument('--flip_data', action="store_true", default=False, help='Augmentation: Flips Input Data')
parser.add_argument('-scale','--scale_data',type=str,choices=['none','minmax','standard'],default='minmax',help="Scale data with minmax(default), normalization or no scaling")
parser.add_argument('-pp','--suppress_preprocess', action="store_true", default=False, help='Augmentation: Suppress Data Processing Block')
parser.add_argument('-ns','--num_sym', type=int, default=0, help='Augmentation: Symbolize Data (default=0, no symbolize)')

parser.add_argument('--use_cuda', action="store_true", default=True, help='Use CUDA if available') # Conflicting: Always True
parser.add_argument('-sche_step','--LR_sche_step', type=int, default=1, help='Stepsize for LR scheduler') # For StepLR
parser.add_argument('-sche_gamma','--LR_sche_gamma', type=float, default=1.0, help='Gamma for LR scheduler') # For StepLR

parser.add_argument('-mode','--mode',type=str,default='trainsz', choices=['trainsz','noise'])
parser.add_argument('-algo','--algo',type=str, default='KNN', choices=['KNN','RF','SVM'])


args = parser.parse_args()
print(args)

mode = args.mode
algo = args.algo


# Algo parameters
# KNN
# range_n_neighbors = [1, 2, 5, 10] #look up to n=num_samples, but 10 was always best
# range_n_neighbors = np.arange(3)[1:] # best k = 68,  acc = 53.33%
# range_n_neighbors = np.arange(80)[1:] # best k = 68,  acc = 53.33%

# RF
# numtreerange = [1, 5, 10, 25, 50, 100, 200]
# depthrange = range(1,11)
# numtreerange = np.arange(3)[1:] # len (3-1)
# depthrange = np.arange(3)[1:] # len (3-1)
numtreerange = np.arange(200)[1:] # best = 
depthrange = np.arange(20)[1:] # best = 

#SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
df_shapes = ['ovo','ovr']
gamma_vals = ['scale','auto']
# Cvals = [1, 5, 10, 25, 50, 75, 100, 500, 1000]
# degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Cvals = np.arange(3)[1:]
# degrees = np.arange(3)[1:]
Cvals = np.arange(1000)[1:]
degrees = np.arange(10)[1:]

# t1 = time.time()
if mode == 'trainsz':
    train_size = [40, 30, 20, 10]
    # train_size = [40]
    n_fold = 3

    print("\n==== Train and Test ====")
    mean_acc, std_acc = [], []

    for ts in train_size: # Loop train sizes dim
        acc = [] # Collected acc for each n_folds
        for idx in range(n_fold): # Loop training index, for each n_folds
            # Extract data 
            AFM_kwargs = {
            'mode':args.data, 'winsz':args.winsz, 'scale_data':args.scale_data,
            'supp_process':args.suppress_preprocess, 'seed':idx}

            AFM_train = AFM(**AFM_kwargs, train=True, unlabeled_percent=0, num_train_per_cls=ts)
            AFM_test = AFM(**AFM_kwargs, train=False, scaler=AFM_train.scaler)

            # print("\n==== Training Data ====")
            _X_train = np.array(AFM_train.data)
            _Y_train = np.array(AFM_train.targets)

            # print("\n==== Testing Data ====")
            X_test = AFM_test.data
            Y_test = AFM_test.targets

            if algo == 'KNN':
                # acc_, best_n_neighbors = knn_(range_n_neighbors, _X_train, _Y_train, X_test, Y_test)
                acc_, best_n_neighbors = knn_(_X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

            if algo == 'RF':
                acc_, best_depth, best_num_trees, seed = rf_(depthrange, numtreerange, _X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

            if algo == 'SVM':
                acc_, best_C, best_kernel, best_degree, best_gamma, best_df_shape, seed  = svm_(Cvals, kernels, degrees, gamma_vals, df_shapes, _X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    mean_acc = np.array(mean_acc)
    std_acc = np.array(std_acc)

    print("mean_acc: ",mean_acc)
    print("std_acc: ",std_acc)

    results = {}
    results["train_size"] = train_size
    results["mean_acc"] = list(mean_acc)
    results["std_acc"] = list(std_acc)

    with open('trainsz_resu/'+algo+'results.json', 'w') as fp1:
        json.dump(results, fp1, indent=4, sort_keys=False)

    t = np.array(train_size) # X ticks

    # Std Dev Plot
    fig, ax = plt.subplots()
    ax.plot(t,mean_acc*100, color="b",label="label")
    ax.fill_between(t, (mean_acc-std_acc)*100, (mean_acc+std_acc)*100, color='b', alpha=.1)
    plt.title(algo+" testing accuracy",fontsize=14)
    plt.xlabel("Training samples per class",fontsize=14)
    plt.ylabel("Test accuracy (%)",fontsize=14)
    # plt.yticks([0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54])
    plt.yticks([35, 40, 45, 50, 55, 60, 65])
    # plt.legend(loc='best')
    # plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(mode+"_resu/"+algo+"results_plot.jpg")

    np.save(mode+'_resu/'+algo+'results_mean_acc.npy',mean_acc)
    np.save(mode+'_resu/'+algo+'results_std_acc.npy',std_acc)

    print("Finished")

if mode == 'noise':
    print("\n==== Noisy Data Percentage ====")
    mean_acc, std_acc = [], []

    train_noise = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # train_noise = [0, 0.1]
    # train_noise = [0]
    n_fold = 3

    for up in train_noise: # Loop train sizes dim
        acc = [] # Collected acc for each n_folds
        for idx in range(n_fold): # Loop training index, for each n_folds
            # Extract training data
            print(f'\n{up*100:.2f}% train noise, Fold {idx}')
            #get data with up 
            AFM_kwargs = {
            'mode':args.data, 'winsz':args.winsz, 'scale_data':args.scale_data,
            'supp_process':args.suppress_preprocess, 'seed':idx}

            AFM_train = AFM(**AFM_kwargs, train=True, unlabeled_percent=up, num_train_per_cls=args.num_train_per_cls)
            AFM_test = AFM(**AFM_kwargs, train=False, scaler=AFM_train.scaler)

            _X_train = AFM_train.data
            _Y_train = AFM_train.targets

            _X_train, _Y_train = shuffle(_X_train, _Y_train, random_state=idx)

            X_test = AFM_test.data
            Y_test = AFM_test.targets

            #Train classifier:

            if algo == 'KNN':
                # clf, acc_, best_n_neighbors = knn_(range_n_neighbors, _X_train, _Y_train, X_test, Y_test)
                acc_, best_n_neighbors = knn_(range_n_neighbors, _X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

                # if up == 0 and idx == 0:
                #     target_names = ['N/MR','SR','DR']
                #     pred = clf.predict(X_test)
                #     clf_rep = classification_report(Y_test,pred, target_names=target_names, digits=6) # For printing

                #     cm = confusion_matrix(Y_test,pred)

                #     print('='*70)
                #     print(clf_rep)
                #     print('Confusion Matrix:')
                #     print(cm)
                #     print('='*70)
                #     print('\n')

            if algo == 'RF':
                # clf, acc_, best_depth, best_num_trees, seed = rf_(depthrange, numtreerange, _X_train, _Y_train, X_test, Y_test)
                acc_, best_depth, best_num_trees, seed = rf_(depthrange, numtreerange, _X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

            if algo == 'SVM':
                # clf, acc_, best_C, best_kernel, best_degree, best_gamma, best_df_shape, seed  = svm_(Cvals, kernels, degrees, gamma_vals, df_shapes, _X_train, _Y_train, X_test, Y_test)
                acc_, best_C, best_kernel, best_degree, best_gamma, best_df_shape, seed  = svm_(Cvals, kernels, degrees, gamma_vals, df_shapes, _X_train, _Y_train, X_test, Y_test)
                acc.append(acc_)

                # if up == 0 and idx == 0:
                #     target_names = ['N/MR','SR','DR']
                #     pred = clf.predict(X_test)
                #     print(Y_test)
                #     print(pred)
                #     clf_rep = classification_report(Y_test,pred, target_names=target_names, digits=6) # For printing

                #     cm = confusion_matrix(Y_test,pred)

                #     print('='*70)
                #     print(clf_rep)
                #     print('Confusion Matrix:')
                #     print(cm)
                #     print('='*70)
                #     print('\n')

        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    mean_acc = np.array(mean_acc)
    std_acc = np.array(std_acc)

    print("mean_acc: ",mean_acc)
    print("std_acc: ",std_acc)
    # breakpoint()

    results = {}
    results["train_noise"] = train_noise
    results["mean_acc"] = list(mean_acc)
    results["std_acc"] = list(std_acc)

    with open('noise_resu/'+algo+'results.json', 'w') as fp1:
        json.dump(results, fp1, indent=4, sort_keys=False)

    t = np.array(train_noise) # X ticks

    # Std Dev Plot
    fig, ax = plt.subplots()
    ax.plot(t,mean_acc*100, color="b",label="label")
    ax.fill_between(t, (mean_acc-std_acc)*100, (mean_acc+std_acc)*100, color='b', alpha=.1)
    plt.title(algo+" testing accuracy",fontsize=14)
    plt.xlabel("Percentage of noisy data (%)",fontsize=14)
    plt.ylabel("Test accuracy (%)",fontsize=14)
    # plt.yticks([0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54])
    plt.yticks([35, 40, 45, 50, 55, 60, 65])
    # plt.legend(loc='best')
    # plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(mode+"_resu/"+algo+"results_plot.jpg")

    np.save(mode+'_resu/'+algo+'results_mean_acc.npy',mean_acc)
    np.save(mode+'_resu/'+algo+'results_std_acc.npy',std_acc)

    print("Finished")

# time_taken = datetime.timedelta(seconds=time.time() - t1)
# print('Time taken: %s s' % (time_taken))
