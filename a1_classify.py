from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import csv
import argparse
import time


def accuracy(C):
    """ Compute accuracy given Numpy array confusion matrix C. Returns a floating point value """
    a_numerator = C.diagonal().sum()
    a_denominator = C.sum()
    if a_denominator == 0:
        acc = 0.0
    else:
        acc = a_numerator / a_denominator
    print("Accuracy computation DONE")
    return acc


def recall(C):
    """ Compute recall given Numpy array confusion matrix C. Returns a list of floating point values """
    rec = []
    # based on j <-> col
    j = 0
    C_transpose = C.T
    for k in C_transpose:
        # transpose to sum up columns
        r_numerator = k[j]
        r_denominator = k.sum()
        if r_denominator == 0:
            rec.append(0.0)
        else:
            rec.append(r_numerator / r_denominator)
        j += 1
    print("Recall computation DONE")
    return rec


def precision(C):
    """ Compute precision given Numpy array confusion matrix C. Returns a list of floating point values """
    pre = []
    # based on i <-> row
    i = 0
    for k in C:
        p_numerator = k[i]
        p_denominator = k.sum()
        if p_denominator == 0:
            pre.append(0.0)
        else:
            pre.append(p_numerator / p_denominator)
        i += 1
    print("Precision computation DONE")
    return pre
    

def class31(filename):
    """" This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    """
    # load data
    feats = np.load(filename)
    data = feats[feats.files[0]]

    # classifiers names
    names = ["LinearSVC",
             "SVC",
             "RandomForestClassifier",
             "MLPClassifier",
             "AdaBoostClassifier"]

    # all classifiers
    classifiers = [
        LinearSVC(max_iter=10000),
        SVC(gamma=2, max_iter=10000),
        RandomForestClassifier(max_depth=5),
        MLPClassifier(alpha=0.05),
        AdaBoostClassifier()]

    # pre-process dataset, split into training and test part
    X, y = data, data
    X_train, X_test, y_train, y_test = train_test_split(X[:, 0:173], y[:, -1], train_size=0.8, test_size=0.2)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # all accuracy
    all_accuracy = []
    # accuracy be key and the iterator i be the value
    iBest_dict = {}
    # iterator i, be the index or the start of each model
    i = 1

    # csv adding helper list
    sub_adder31 = []
    full_adder31 = []

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        # generate title
        print(str(i) + ": " + name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        accu = accuracy(c_matrix)
        print(accu)
        recc = recall(c_matrix)
        print(recc)
        prec = precision(c_matrix)
        print(prec)

        # add accuracy paired with the i into dict
        iBest_dict[accu] = i
        # add accuracy into a list and use to comparing at the end
        all_accuracy.append(accu)

        # write in .csv file with appropriate format line by line
        if i <= 5:
            sub_adder31.append(i)
            sub_adder31.append(accu)
            for num in recc:
                sub_adder31.append(num)
            for num in prec:
                sub_adder31.append(num)
            for i_row in c_matrix:
                for j_col in i_row:
                    sub_adder31.append(j_col)
            full_adder31.append(sub_adder31)
            sub_adder31 = []
        else:
            break
        i += 1

        print("\n---------------------------------------------------------------\n")

    # write into a .csv file
    myFile = open('a1_3.1.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(full_adder31)

    # find the BEST model based on their accuracy
    print("All accuracy are: %s" % all_accuracy)
    max_acc = max(all_accuracy)
    iBest = iBest_dict[max_acc]
    print("The best model number is %s" % iBest)

    return X_train, X_test, y_train, y_test, iBest


def class32(X_train, X_test, y_train, y_test, iBest):
    """ This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    """
    clf = None

    # select classifiers
    if iBest == 1:
        clf = LinearSVC(max_iter=10000)
        print("Choosing LinearSVC")
    elif iBest == 2:
        clf = SVC(gamma=2, max_iter=10000)
        print("Choosing SVC")
    elif iBest == 3:
        clf = RandomForestClassifier(max_depth=5)
        print("Choosing RandomForestClassifier")
    elif iBest == 4:
        clf = MLPClassifier(alpha=0.05)
        print("Choosing MLPClassifier")
    elif iBest == 5:
        clf = AdaBoostClassifier()
        print("Choosing AdaBoostClassifier")

    # csv adder
    adder32 = [[]]

    # preprocess dataset, split into training and test part
    # training data amount is 1k
    X_1k_train, X_1k_test, y_1k_train, y_1k_test = train_test_split(X_train, y_train, train_size=0.03125, test_size=0.96875)
    clf.fit(X_1k_train, y_1k_train)
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_1k = accuracy(c_matrix)
    adder32[0].append(accu_1k)
    print("Accuracy of 1k is %s" % accu_1k)
    print("\n---------------------------------------------------------------\n")

    # training data amount is 5k
    X_5k_train, X_5k_test, y_5k_train, y_5k_test = train_test_split(X_train, y_train, train_size=0.15625, test_size=0.84375)
    clf.fit(X_5k_train, y_5k_train)
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_5k = accuracy(c_matrix)
    adder32[0].append(accu_5k)
    print("Accuracy of 5k is %s" % accu_5k)
    print("\n---------------------------------------------------------------\n")

    # training data amount is 10k
    X_10k_train, X_10k_test, y_10k_train, y_10k_test = train_test_split(X_train, y_train, train_size=0.3125, test_size=0.6875)
    clf.fit(X_10k_train, y_10k_train)
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_10k = accuracy(c_matrix)
    adder32[0].append(accu_10k)
    print("Accuracy of 10k is %s" % accu_10k)
    print("\n---------------------------------------------------------------\n")

    # training data amount is 15k
    X_15k_train, X_15k_test, y_15k_train, y_15k_test = train_test_split(X_train, y_train, train_size=0.46875, test_size=0.53125)
    clf.fit(X_15k_train, y_15k_train)
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_15k = accuracy(c_matrix)
    adder32[0].append(accu_15k)
    print("Accuracy of 15k is %s" % accu_15k)
    print("\n---------------------------------------------------------------\n")

    # training data amount is 20k
    X_20k_train, X_20k_test, y_20k_train, y_20k_test = train_test_split(X_train, y_train, train_size=0.625, test_size=0.375)
    clf.fit(X_20k_train, y_20k_train)
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_20k = accuracy(c_matrix)
    adder32[0].append(accu_20k)
    print("Accuracy of 20k is %s" % accu_20k)
    print("\n---------------------------------------------------------------\n")

    # write into a .csv file
    myFile = open('a1_3.2.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(adder32)

    # change X_1k value and y_1k value, communicate 3.2 with 3.3
    X_1k, y_1k = X_1k_train, y_1k_train

    return X_1k, y_1k


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    """ This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    """
    clf = None

    # select classifiers
    if i == 1:
        clf = LinearSVC(max_iter=10000)
        print("Choosing LinearSVC")
    elif i == 2:
        clf = SVC(gamma=2, max_iter=10000)
        print("Choosing SVC")
    elif i == 3:
        clf = RandomForestClassifier(max_depth=5)
        print("Choosing RandomForestClassifier")
    elif i == 4:
        clf = MLPClassifier(alpha=0.05)
        print("Choosing MLPClassifier")
    elif i == 5:
        clf = AdaBoostClassifier()
        print("Choosing AdaBoostClassifier")

    # csv adder
    line_1_6 = []
    line_7 = []
    line_8_10 = []

    np.seterr(divide='ignore', invalid='ignore')
    # 1k from 3.2
    for k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(f_classif, k)
        selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_
        if k == 5:
            index_1k_k_is_5 = [i for i in np.nditer(selector.get_support(indices=True))]
            best_1k_5_featspp = [pp[i] for i in np.nditer(selector.get_support(indices=True))]
        best_k_feats = [k] + [pp[i] for i in np.nditer(selector.get_support(indices=True))]
        print("1k from 3.2 and its " + str(k) + " best features' p-value:")
        print(best_k_feats)

    # 32k from 3.1
    for k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(f_classif, k)
        selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        if k == 5:
            index_32k_k_is_5 = [i for i in np.nditer(selector.get_support(indices=True))]
            best_32k_5_featspp = [pp[i] for i in np.nditer(selector.get_support(indices=True))]
        best_k_feats = [k] + [pp[i] for i in np.nditer(selector.get_support(indices=True))]
        print("32k from 3.1 and its " + str(k) + " best features' p-value:")
        print(best_k_feats)
        line_1_6.append(best_k_feats)

    # selector is with k=5 attribute
    selector = SelectKBest(f_classif, k=5)

    # train classifier with amount 1k
    X_train_1k = selector.fit_transform(X_1k, y_1k)
    X_test_1k = selector.transform(X_test)
    clf.fit(X_train_1k, y_1k)
    y_pred = clf.predict(X_test_1k)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_1k = accuracy(c_matrix)
    print("Accuracy of 1k is %s" % accu_1k)

    # train classifier with amount 32k
    X_train_32k = selector.fit_transform(X_train, y_train)
    X_test_32k = selector.transform(X_test)
    clf.fit(X_train_32k, y_train)
    y_pred = clf.predict(X_test_32k)
    c_matrix = confusion_matrix(y_test, y_pred)
    accu_32k = accuracy(c_matrix)
    print("Accuracy of 32k is %s" % accu_32k)

    double_accu = [accu_1k, accu_32k]
    line_7.append(double_accu)
    line_8_10.append(index_1k_k_is_5)
    line_8_10.append(best_1k_5_featspp)
    line_8_10.append(index_32k_k_is_5)
    line_8_10.append(best_32k_5_featspp)

    # write into a .csv file
    myFile = open('a1_3.3.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(line_1_6)
        writer.writerows(line_7)
        # suppose we analyze the top five features' index and their p-val...
        writer.writerows(line_8_10)


def class34(filename, i):
    """ This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
    """
    # load data
    feats = np.load(filename)
    data = feats[feats.files[0]]

    # classifiers names
    names = ["LinearSVC",
             "SVC",
             "RandomForestClassifier",
             "MLPClassifier",
             "AdaBoostClassifier"]

    # all classifiers
    classifiers = [
        LinearSVC(max_iter=10000),
        SVC(gamma=2, max_iter=10000),
        RandomForestClassifier(max_depth=5),
        MLPClassifier(alpha=0.05),
        AdaBoostClassifier()]

    # pre-process dataset, split into training and test part, same as 3.1
    # split by given all the initially available data.
    X, y = data[:, 0:173], data[:, -1]

    # adder helper, add data line by line into the .csv file
    kfold_classifier_row = []
    kfold_classifier_matrix = []
    p_val = [[]]

    # setting 5-fold cross-validation for kfold funciton
    kfold = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # index of model
        index = 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            # generate title
            print(str(index) + ": " + name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            c_matrix = confusion_matrix(y_test, y_pred)
            accu = accuracy(c_matrix)
            print(accu)
            kfold_classifier_row.append(accu)
            index += 1

        kfold_classifier_matrix.append(kfold_classifier_row)
        kfold_classifier_row = []

    kfold_classifier_matrix = np.array(kfold_classifier_matrix)
    # print(kfold_classifier_matrix)
    transpose = kfold_classifier_matrix.T
    # print(transpose)

    for kBest in range(1, 6):
        # here i is the same as the iBest
        if i != kBest:
            print(str(i) + " with " + str(kBest))
            # print(transpose[i - 1])
            # print(transpose[kBest - 1])
            S, pval = stats.ttest_rel(transpose[i - 1], transpose[kBest - 1])
            p_val[0].append(pval)

    # write into a .csv file
    myFile = open('a1_3.4.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(kfold_classifier_matrix)
        writer.writerows(p_val)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each.')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    whole_start_time = time.time()

    # TODO : complete each classification experiment, in sequence.
    start_time = time.time()
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    print("Section 3.1 DONE")
    print("--- Use %s seconds to finish class31 ---" % (time.time() - start_time))
    print("\n***************************************************************\n")

    start_time = time.time()
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    print("Section 3.2 DONE")
    print("--- Use %s seconds to finish class32 ---" % (time.time() - start_time))
    print("\n***************************************************************\n")

    start_time = time.time()
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    print("Section 3.3 DONE")
    print("--- Use %s seconds to finish class33 ---" % (time.time() - start_time))
    print("\n***************************************************************\n")

    start_time = time.time()
    class34(args.input, iBest)
    print("Section 3.4 DONE")
    print("--- Use %s seconds to finish class34 ---" % (time.time() - start_time))
    print("\n***************************************************************\n")

    print("PART 3 DONE")
    print("--- Use %s seconds to finish PART3 ---" % (time.time() - whole_start_time))

    # PART 3 DONE
    # --- Use 6969.061643123627 seconds to finish PART3 ---
    # About 116 min
