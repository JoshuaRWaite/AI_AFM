from numpy.random import seed 
seed(4) # fix seed for result reproducibility
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import json
import random
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import numpy as np
from copy import deepcopy

# def knn_(range_n_neighbors, _X_train, _Y_train, X_test, Y_test):
def knn_(_X_train, _Y_train, X_test, Y_test):
    range_n_neighbors = np.arange(len(_Y_train))[1:]
    best_testacc = 0
    best_n_neighbors = 1
    for n_neighbors in range_n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        # print(f'Training kNN with k = {n_neighbors}')
        clf.fit(_X_train, _Y_train)
        testacc = clf.score(X_test,Y_test)

        if testacc > best_testacc:
            best_testacc = testacc
            best_n_neighbors = n_neighbors
            # best_clf = deepcopy(clf)
    print(f'\nBest Test Acc: {best_testacc*100:.2f} %')
    print(f'Best k: {best_n_neighbors}')
    # return best_clf, best_testacc, best_n_neighbors
    return best_testacc, best_n_neighbors

def rf_(depthrange, numtreerange, _X_train, _Y_train, X_test, Y_test):
    best_testacc = 0
    best_depth = 1
    best_num_trees = 1
    best_seed = 0

    for depth in depthrange:
        for num_trees in numtreerange:
            for seed in np.arange(3): # 3 fold seed
                clf = RandomForestClassifier(bootstrap=False,n_estimators=num_trees,max_features=None,criterion='gini',max_depth=depth, random_state=seed, n_jobs=-1)
                # print(f'Training RF with d = {depth}')
                # print(f'Training RF with n = {num_trees}')
                # print(f'Training RF with s = {seed}')
                clf.fit(_X_train, _Y_train)
                testacc = clf.score(X_test,Y_test)

                if testacc > best_testacc:
                    best_testacc = testacc
                    best_depth = depth
                    best_num_trees = num_trees
                    best_seed = seed
                    # best_clf = deepcopy(clf)

    print(f'\nBest Test Acc: {best_testacc*100:.2f} %')
    print(f'Best depth: {best_depth}')
    print(f'Best num trees: {best_num_trees}')
    print(f'Best seed: {best_seed}')
    # return best_clf, best_testacc, best_depth, best_num_trees, best_seed
    return best_testacc, best_depth, best_num_trees, best_seed


def svm_(Cvals, kernels, degrees, gamma_vals, df_shapes, _X_train, _Y_train, X_test, Y_test):
    best_C = 1
    best_gamma = 0
    best_testacc = 0
    best_kernel = 'linear'
    best_degree = 0
    best_gamma = 'scale'
    best_df_shape = 'ovo'
    best_seed = 0

    for kernel in kernels:
        for degree in degrees:
            for c in Cvals:
                for gamma in gamma_vals:
                    for df_shape in df_shapes:
                        for seed in np.arange(3): # 3 fold seed
                            clf=svm.SVC(C=c, kernel=kernel, decision_function_shape=df_shape, degree=degree, gamma=gamma, random_state=seed)
                            clf.fit(_X_train, _Y_train)
                            testacc = clf.score(X_test,Y_test)

                            if testacc > best_testacc:
                                best_testacc = testacc
                                best_C = c
                                best_kernel = kernel
                                best_degree = degree
                                best_gamma = gamma
                                best_df_shape = df_shape
                                best_seed = seed
                                # best_clf = deepcopy(clf)

    print(f'\nBest Test Acc: {best_testacc*100:.2f} %')
    print(f'Best C: {best_C}')
    print(f'Best kernel: {best_kernel}')
    print(f'Best deg: {best_degree}')
    print(f'Best gamma: {best_gamma}')
    print(f'Best shape: {best_df_shape}')
    print(f'Best seed: {best_seed}')
    # return best_clf, best_clf, best_testacc, best_C, best_kernel, best_degree, best_gamma, best_df_shape, best_seed
    return best_testacc, best_C, best_kernel, best_degree, best_gamma, best_df_shape, best_seed
