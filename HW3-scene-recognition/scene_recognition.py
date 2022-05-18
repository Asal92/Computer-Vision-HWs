import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg') # My MacOSX backend won't open plots!! so instead I force it to use another backend
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from scipy import stats
from pathlib import Path, PureWindowsPath

TINY_IMAGE_SIZE = (16,16)
CONFUSION_MATRIX_SIZE = (15,15)
KNN_DICT_SIZE = 50
SVM_DICT_SIZE = 70
STRIDE = 16
PATCH_SIZE = 16

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def get_tiny_image(img, output_size):
    '''
    :param img: gray scale image
    :param output_size: wxh (256=16x16) of tiny image
    :return: feature is a flattened vector with mean 0 and unit length with size 1x256
    '''
    # resizing and flattening the image
    resized_img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    resized_img = resized_img.flatten()
    # mean_zero by subtracting from mean
    mean = np.mean(resized_img)
    mean_zero_img = resized_img - mean
    # normalizing by dividing by norm = unit length
    norm = np.linalg.norm(mean_zero_img)
    tiny_img = mean_zero_img / norm

    #cv2.imshow("img", tiny_img)
    #cv2.waitKey()
    feature = tiny_img

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    '''
    :param feature_train: ntr x d (training samples , d is dimension of image feature 256=16x16)
    :param label_train: vector [1,15]
    :param feature_test: nte x d (testing samples)
    :param k: number of neighbors for prediction
    :return: list of predictions- easy peasy
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(feature_train, label_train)
    label_test_pred = knn.predict(feature_test)

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    '''
    :param label_classes: list of all classes [1,15]
    :param label_train_list: corrresponding labels
    :param img_train_list: path to training samples
    :param label_test_list: corrresponding labels
    :param img_test_list: path to testing samples
    :return: 15 x 15 confusion matrix , accuracy of testing prediction
    Algorithm:
    1. load training and testing images
    2. build image representation
    3. train a classifier using the representation of the training images
    4. classify the testing data = KNN
    5. compute accuracy of the testing data classification >> accuracy > 18%
    '''
    train_feature = np.empty((0, 256))
    test_feature = np.empty((0, 256))

    # let's open training and testing samples, get their tiny images and make that n x d matrix
    print("getting tiny images of training list")
    for ntr in img_train_list:
        img = cv2.imread(ntr, 0) # recall 0 is for gray scale
        tiny_img = get_tiny_image(img, TINY_IMAGE_SIZE)
        train_feature = np.vstack((train_feature, tiny_img))

    print("getting tiny images of testing list")
    for nte in img_test_list:
        img = cv2.imread(nte, 0)
        tiny_img = get_tiny_image(img, TINY_IMAGE_SIZE)
        test_feature = np.vstack((test_feature, tiny_img))

    train_feature_array = np.asarray(train_feature)
    test_feature_array = np.asarray(test_feature)


    # now I have to send feature_train and feature_test to KNN predict and see how much of it is correct
    print("predicting labels for test set using KNN")
    # changing labels from strings to [1,15]
    label_train_int = [label_classes.index(ntr) + 1 for ntr in label_train_list]
    # Make sure to use ndarray instead of list for predict_knn or predict_svm!!
    label_test_pred_int = predict_knn(train_feature_array, label_train_int, test_feature_array, k=10)
    # changing predictions from [1,15] to the correct labels
    label_test_pred = [label_classes[c - 1] for c in label_test_pred_int]

    # to count how many samples of each class I have for confusion matrix
    class_total = {}
    for c in label_classes:
        class_total[c] = sum(1 for i in label_test_list if i==c)

    #make a confusion matrix
    print("building confusion matrix")
    confusion_matrix = np.zeros(CONFUSION_MATRIX_SIZE)
    for i in range(len(label_classes)):
        for j in range(len(label_classes)):
            ii = label_classes[i] # rows
            jj = label_classes[j] # columns
            cc = class_total[label_classes[i]]
            s = sum(1 for n in range(len(label_test_pred)) if (label_test_list[n] == ii) & (label_test_pred[n] == jj))
            confusion_matrix[i][j] = s/cc

    # accuracy = mean of correct predictions
    accu = 0
    for i in range(len(label_classes)):
        accu += confusion_matrix[i][i]
    mean_accu = accu / len(label_classes)

    confusion = confusion_matrix
    accuracy = mean_accu
    print(f'Tiny Image + KNN accuracy is {accuracy}')

    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy



def compute_dsift(img, stride, size):
    '''
    :param img: gray scale image
    :param stride: how many pixels to move, where keypoints are
    :param size: size of keypoint diameter
    :return: dense_feature , collection of sift descriptors nx128
    we make local patches using stride, stride means how many pixels we wanna move
    then we give location of that pixel to compute descriptor for.
    when we call sift to compute descriptor, we should give it the keypoint location and size of keypoint
    '''
    sift = cv2.SIFT_create()
    # let's see how much we can move and how many patches we can make
    h, w = np.shape(img)
    w_step = int((w - size) / stride) + 1 # int: make sure it's not float, +1 to include all steps
    h_step = int((h - size) / stride) + 1
    keyPoints = []
    # now we will make our local patches and make a list of our keypoints
    for i in range(h_step):
        for j in range(w_step):
            kp_diameter = size
            kp_y = (i + 1) * stride
            kp_x = (j + 1) * stride
            if kp_y > h or kp_x > w:
                continue # make sure keypoints are not out of range
            keyPoints.append(cv2.KeyPoint(x=kp_x, y=kp_y, size=kp_diameter))
    # patch_kp, patch_des = sift.detectAndCompute(img, None)
    patch_kp, patch_des = sift.compute(img,keyPoints)
    dense_feature = patch_des
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size):
    '''
    :param dense_feature_list: list of dense features for training images
    :param dic_size: size of the dictionary?! number of visual words, is it number of clusters?
    :return:visual words with size dic_size x 128
    1.computer dense sift for each image,
    2. build a pool of sift features from all training images
    3. cluter using kmeans algorithm
    4. return cluster centers
    '''

    clusters = KMeans(n_clusters=dic_size, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                      random_state=0, copy_x=True, algorithm='auto')
    clusters.fit(dense_feature_list) # finding clusters for training set
    vocab = clusters.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    '''
    :param feature: set of sift features for <one image>
    :param vocab: visual dictionary from build_visual_dictionary from training set
    :return: bow_feature bag of words feature size is dic_size
    >> use nearest neighbors to find closest cluster
    >> normalize the histogram at the end
    >> make sure bow_feature is the size of dic_size
    '''
    dict_size = vocab.shape[0]
    clusters_indx = [i for i in range(dict_size)]
    bow_feature = [0 for i in range(dict_size)]

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(vocab, clusters_indx)
    predictions = neigh.predict(feature)

    for p in predictions:
        bow_feature[p] += 1

    # normalizing the bow_feature
    bow_feature = np.asarray(bow_feature)
    norm = np.linalg.norm(bow_feature)
    bow_feature = bow_feature / norm

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    '''
    :param label_classes: list of all classes [1,15]
    :param label_train_list: corrresponding labels
    :param img_train_list: path to training samples
    :param label_test_list: corrresponding labels
    :param img_test_list: path to testing samples
    :return: 15 x 15 confusion matrix , accuracy of testing prediction
    Algorithm:
    1. load training and testing images
    2. build image representation = combine build_visual_dictionary and compute_bow
    3. train a classifier using the representation of the training images
    4. classify the testing data = KNN
    5. compute accuracy of the testing data classification >> accuracy > 50%
    '''

    train_dense_list, test_dense_list = [], []
    train_dense = np.empty((0, 128))  # bc sift descriptor is 128 dimensional
    test_dense = np.empty((0, 128))
    train_bow = np.empty((0,KNN_DICT_SIZE)) # bc bow is size of dic_size
    test_bow = np.empty((0, KNN_DICT_SIZE))

    print("computing sift for training set")
    for ntr in img_train_list:
        img = cv2.imread(ntr, 0)  # recall 0 is for gray scale
        dsift = compute_dsift(img, STRIDE, PATCH_SIZE)
        train_dense = np.vstack((train_dense, dsift))
        train_dense_list.append(dsift)#couldn't stack them all, bc I need all sifts of one image to pass to compute_bow

    print("computing sift for testing set")
    for nte in img_test_list:
        img = cv2.imread(nte, 0)  # recall 0 is for gray scale
        dsift = compute_dsift(img, STRIDE, PATCH_SIZE)
        test_dense = np.vstack((test_dense, dsift))
        test_dense_list.append(dsift)

    # now we need to build a dictionary based on sifts
    print("building dictionary")
    train_vocab = build_visual_dictionary(train_dense, KNN_DICT_SIZE)
    np.savetxt('vocab.txt', train_vocab)
    #train_vocab = np.loadtxt('vocab.txt')

    print("computing bow for training set")
    for dtr in train_dense_list:
        bow = compute_bow(dtr, train_vocab)
        train_bow = np.vstack((train_bow, bow))

    print("computing bow for testing set")
    for dte in test_dense_list:
        bow = compute_bow(dte, train_vocab)
        test_bow = np.vstack((test_bow, bow))

    print("predicting labels for test set using KNN")
    # changing labels from strings to [1,15]
    label_train_int = [label_classes.index(ntr) + 1 for ntr in label_train_list]
    label_test_pred_int = predict_knn(train_bow, label_train_int, test_bow, k=15)
    # changing predictions from [1,15] to the correct labels
    label_test_pred = [label_classes[c - 1] for c in label_test_pred_int]

    # to count how many samples of each class I have for confusion matrix
    class_total = {}
    for c in label_classes:
        class_total[c] = sum(1 for i in label_test_list if i == c)

    print("building confusion matrix")
    confusion_matrix = np.zeros(CONFUSION_MATRIX_SIZE)
    for i in range(len(label_classes)):
        for j in range(len(label_classes)):
            ii = label_classes[i]  # rows
            jj = label_classes[j]  # columns
            cc = class_total[label_classes[i]]
            s = sum(
                1 for n in range(len(label_test_pred)) if (label_test_list[n] == ii) & (label_test_pred[n] == jj))
            confusion_matrix[i][j] = s / cc

    # accuracy = mean of correct predictions
    accu = 0
    for i in range(len(label_classes)):
        accu += confusion_matrix[i][i]
    mean_accu = accu / len(label_classes)

    confusion = confusion_matrix
    accuracy = mean_accu
    print(f'BoW + KNN accuracy is {accuracy}')

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    ''' ok before we classified using KNN , now we wanna use SVM as a classifier
    but we have to do one-vs-all SVM. meaning I have to have 15 different svm classifier?
    :param feature_train: ntr x d(dimension of images) each row is the feature of training sample
    :param label_train: [1,15]
    :param feature_test: nte x d (features of testing set)
    :param n_classes: No explanation in the HW!!!! I will assume it's the number of class labels
    :return: nte vector with predictions using svm classifier
    from hw: all 15 classifiers will be evaluated on each test case , and the classifier which is most
    confidently positive wins!!!
    So I have to make 15 svm models, fit them with one vs rest decision function
    then predict feature tests 15 times and pick the class with highest probability.
    '''
    svm_models, svm_preds = [], []
    for i in range(n_classes):
        '''svm = SVC( C=3.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                            tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1,
                            decision_function_shape='ovr', break_ties=False, random_state=None)'''
        svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=4.0, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                        max_iter=1000)
        svm_models.append(svm)
        c = i+1
        ovr_class = [1 if l==c else 0 for l in label_train] # one-vs-rest
        svm.fit(feature_train, ovr_class)

    for i in range(n_classes):
        #svm_preds.append(list(svm_models[i].predict_proba(feature_test)[:,1]))
        svm_preds.append(svm_models[i].decision_function(feature_test))

    # this returns the class of max value for each column
    svm_pred_indx = np.argmax(svm_preds, axis=0)
    svm_pred_indx += 1 # currently is [0,14] and should be [1,15]
    label_test_pred = svm_pred_indx
    return label_test_pred



def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    '''
    :param label_classes: list of all classes [1,15]
    :param label_train_list: corrresponding labels
    :param img_train_list: path to training samples
    :param label_test_list: corrresponding labels
    :param img_test_list: path to testing samples
    :return: 15 x 15 confusion matrix , accuracy of testing prediction
    Algorithm:
    1. load training and testing images
    2. build image representation = combine build_visual_dictionary and compute_bow
    3. train a classifier using the representation of the training images
    4. classify the testing data = KNN
    5. compute accuracy of the testing data classification >> accuracy > 60%
    '''

    train_dense_list, test_dense_list = [], []
    train_dense = np.empty((0, 128))  # bc sift descriptor is 128 dimensional
    test_dense = np.empty((0, 128))
    train_bow = np.empty((0, SVM_DICT_SIZE))  # bc bow is size of dic_size
    test_bow = np.empty((0, SVM_DICT_SIZE))

    print("computing sift for training set")
    for ntr in img_train_list:
        img = cv2.imread(ntr, 0)  # recall 0 is for gray scale
        dsift = compute_dsift(img, STRIDE, PATCH_SIZE)
        train_dense = np.vstack((train_dense, dsift))
        train_dense_list.append(dsift)#couldn't stack them all, bc I need all sifts of one image to pass to compute_bow
    #np.savetxt('train_dense.txt', train_dense)

    print("computing sift for testing set")
    for nte in img_test_list:
        img = cv2.imread(nte, 0)  # recall 0 is for gray scale
        dsift = compute_dsift(img, STRIDE, PATCH_SIZE)
        test_dense = np.vstack((test_dense, dsift))
        test_dense_list.append(dsift)
    #np.savetxt('test_dense.txt', test_dense)

    # now we need to build a dictionary based on sifts
    print("building dictionary")
    train_vocab = build_visual_dictionary(train_dense, SVM_DICT_SIZE)
    #np.savetxt('vocab.txt', train_vocab)
    #train_vocab = np.loadtxt('vocab.txt')

    print("computing bow for training set")
    for dtr in train_dense_list:
        bow = compute_bow(dtr, train_vocab)
        train_bow = np.vstack((train_bow, bow))

    print("computing bow for testing set")
    for dte in test_dense_list:
        bow = compute_bow(dte, train_vocab)
        test_bow = np.vstack((test_bow, bow))

    print("predicting labels for test set using KNN")
    # changing labels from strings to [1,15]
    label_train_int = [label_classes.index(ntr) + 1 for ntr in label_train_list]
    label_test_pred_int = predict_svm(train_bow, label_train_int, test_bow, len(label_classes))
    # changing predictions from [1,15] to the correct labels
    label_test_pred = [label_classes[c - 1] for c in label_test_pred_int]

    # to count how many samples of each class I have for confusion matrix
    class_total = {}
    for c in label_classes:
        class_total[c] = sum(1 for i in label_test_list if i == c)

    print("building confusion matrix")
    confusion_matrix = np.zeros(CONFUSION_MATRIX_SIZE)
    for i in range(len(label_classes)):
        for j in range(len(label_classes)):
            ii = label_classes[i]  # rows
            jj = label_classes[j]  # columns
            cc = class_total[label_classes[i]]
            s = sum(
                1 for n in range(len(label_test_pred)) if (label_test_list[n] == ii) & (label_test_pred[n] == jj))
            confusion_matrix[i][j] = s / cc

    # accuracy = mean of correct predictions
    accu = 0
    for i in range(len(label_classes)):
        accu += confusion_matrix[i][i]
    mean_accu = accu / len(label_classes)

    confusion = confusion_matrix
    accuracy = mean_accu
    print(f'BoW + SVM accuracy is {accuracy}')

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy

def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    '''label classes = list of 15 classes
    label train list = list of labels of train list
    img train list = list of image names of train list
    label test list = list of labels of test list
    img test list = list of image names of test list'''
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)



