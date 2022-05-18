import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

IMAGE_SIZE = 196
NUM_CLASSES = 10
NUM_HID_LAYERS = 30
CONV_SIZE = 3
FLAT_LAYER = 147
POOLING_STRIDE = 2

def one_hot_encoding(labels):
    '''
    convert the labels into one-hot vector for training
    label dim is (1,12000)  , labels should be 0-9
    '''
    one_hot = np.zeros([10, len(labels[0])])
    for i in range(len(labels[0])):
        l = labels[0,i]
        one_hot[l,i] = 1
    return one_hot

def visualize_loss(loss_list):
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(loss_list)
    plt.show()


def get_mini_batch(im_train, label_train, batch_size):
    ''' shuffle the train samples abd make a list of batches
    :param im_train: 196 x 12000
    :param label_train: 1 x 12000
    :param batch_size: 32
    :return: batches of samples and their labels that are shuffled. batch_x: 196 x batch_size batch_y: 10 x batch size
    '''
    mini_batch_x, mini_batch_y = [], []
    train_size = len(im_train[0])
    batch_int = train_size // batch_size  # how many batches we will have
    batch_remainder = train_size % batch_size
    train_random= list(np.arange(train_size))
    np.random.shuffle(train_random)

    one_hot_labels = one_hot_encoding(label_train)

    batch = 0
    for i in range(batch_int): #375
        batch_train, batch_label = [], []
        for b in range(batch_size): # 32
            batch_train.append(im_train[:, train_random[batch + b]])
            batch_label.append(one_hot_labels[:, train_random[batch + b]])
        batch += batch_size
        mini_batch_x.append(np.transpose(batch_train))
        mini_batch_y.append(np.transpose(batch_label))

    # now if batch has less than batch_size samples:
    if batch_remainder > 0:
        for b in range(batch_remainder):
            batch_train.append(im_train[:, train_random[batch + b]])
            batch_label.append(one_hot_labels[:, train_random[batch + b]])
        mini_batch_x.append(np.transpose(batch_train))
        mini_batch_y.append(np.transpose(batch_label))

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    ''' Fully Connected Layer
    :param x: mx1 input to FC layer
    :param w: weights nxm
    :param b: bias nx1
    :return: y = wx + b  nx1
    '''
    #x = np.reshape((len(x), 1))
    y = np.matmul(w, x) + b
    y = np.reshape(y, (len(y),1))
    return y


def fc_backward(dl_dy, x, w, b, y):
    ''' n -> output dim  m -> input dim
    :param dl_dy: 1xn loss derivative wrt output y
    :return: dl_dx 1xm loos derivative wrt x 1xm
    :return: dl_dw 1 x (nxm)
    :return: dl_db 1xn
    dl_dx = dl_dy.dy_dx = dl_dy . w  [y=wx+b]
    dl_dw = dl_dy.dy_dw = dl_dy . x = 2*(y_tilde - y)*x
    dl_db = dl_dy.dy_db = dl_db . I(Identity)
    '''
    #dl_dy = np.reshape(dl_dy, (len(dl_dy), 1))
    n = len(dl_dy[0])
    m = len(x)
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))

    dl_dx = np.matmul(dl_dy, w) #1xn nxm = 1xm
    #dl_dx = np.reshape(dl_dx, (1, m))

    dl_dw = np.matmul(dl_dy.T, x.T)
    dl_dw = np.ndarray.flatten(dl_dw)
    dl_dw = np.reshape(dl_dw, (1, m*n))

    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    '''
    :param y_tilde: prediction m
    :param y: ground truth {0,1}xm
    :return: l loss=Euclidean distance L = ||y_tilda - y||**2
    :return: dl_dy loss derivative wrt prediction (y~-y)**2  -> dl_dy = 2(y~-y)
    '''
    #y = np.reshape(y, (len(y),1))
    l = np.sum((y - y_tilde) ** 2)
    #l = np.linalg.norm(y - y_tilde)
    dl_dy = (y_tilde - y) * 2
    dl_dy = np.reshape(dl_dy, (1, len(dl_dy))) # 1xn
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    '''
    :param x: mx1 input to softmax (y_tilde)
    :param y: 10x1 {0,1} ground truth
    :return: l=cross entropy loss, dl_dy=loss derivative wrt x
    softmax = ei / sum(ei)
    dL/dy = y~ - y = dl_dytild * dytilde_dy  from lec 25
    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    '''
    l, dl_dy=0,0
    softmax_l = []
    e = np.exp(x)
    for i in range(len(x)):#soft = e/e.sum()
        soft = e[i]/np.sum(e,axis=0)
        softmax_l.append(soft)
    y_tilde = np.asarray(softmax_l)

    # L = sum(gt * log(y_tilde))
    for i in range(len(y)):
        l += (y[i] * np.log(y_tilde[i]))

    dl_dy = y_tilde - np.reshape(y, (len(y), 1)) # -y/ytilde * dl_dy f(1-f) i=j ??
    dl_dy = np.reshape(dl_dy, (1, len(dl_dy))) # 1xn

    return l, dl_dy

def relu(x):
    '''
    :param x: tensor/vector/matrix
    :return: y ReLU y=max(0,x)
    '''
    #m,n = np.shape(x) # output should be the same shape as input
    # it messed it up for cnn when I have more than 2 dimensions
    shape = np.shape(x)
    x_flat = np.ndarray.flatten(x)
    flat_shape = np.shape(x_flat)
    #y = np.zeros((m*n))
    y = np.zeros(flat_shape)
    for i in range(len(x_flat)):
        if x_flat[i] > 0:
            y[i] = x_flat[i]
        else:
            #y[i] = x_flat[i] * 0.01 # Leaky ReLU
            y[i] = x_flat[i] * 0.0
    y = np.reshape(y, shape)

    return y


def relu_backward(dl_dy, x, y):
    '''
    :param dl_dy: loss derivative wrt output y 1xz (z size of input)
    :param x: input (196)
    :param y: output (10)
    :return: loss derivative wrt input x (1xz)
    dl_dx = dl_dy if yi>=0 , 0 otherwise (directly from slides!)
    '''
    #m,n = np.shape(dl_dy)
    shape = np.shape(dl_dy)
    dl_dy_flat = np.ndarray.flatten(dl_dy)
    x_flat = np.ndarray.flatten(x)
    flat_shape = np.shape(x_flat)
    dl_dx = np.zeros(flat_shape)
    for i in range(len(dl_dy_flat)):
        if x_flat[i] > 0:
            dl_dx[i] = dl_dy_flat[i]
        else:  # trying leaky relu to see if it makes my accuracy better!
            # dl_dx[i] = dl_dy_flat[i] * 0.01
            dl_dx[i] = dl_dy_flat[i] * 0.0

    dl_dx = np.reshape(dl_dx, shape)

    return dl_dx


def conv(x, w_conv, b_conv):
    '''
    :param x: H x W x C1 (C1 = 1 ?)
    :param w_conv, b_conv = weights (hxwxC1xC2) and bias(C2x1) of conv operation
    :return: H x W x C2 (C2 = 3 ? )
    >> pad zero at the boundry of the image (np.pad)
    >> im2col to simplify convolutional operation - but its matlab method!!!
    '''
    H, W, C1 = np.shape(x)  # expecting 14x14x1
    h, w, C1, C2 = np.shape(w_conv) # c2 would be output channel
    y = np.zeros((H, W, C2))  # still should have the same size as input except with C2 channels

    # x is 14x14x1 , filter is 3x3, only need to add pads 1 top 1 bottom
    padded_x = np.pad(x, ((1,1), (1,1), (0,0)), 'constant', constant_values=(0, 0)) #image should become 16x16x1 (no pad on 3rd dim)

    # we have to make 3 channels now, so first loop for output channels
    for out_c in range(C2):
        # now do one full convolution
        # remember filter
        for i in range(H): # stride is 1, so no worries here
            for j in range(W):
                y[i][j][out_c] = np.sum(np.multiply(padded_x[i:i + 3, j:j + 3, :], w_conv[:,:,:,out_c]))
                y[i][j][out_c] += b_conv[out_c]
    # copied this from my HW1. this way was easier without im2col
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    '''
    dl_dy=14x14x3 , x=14x14x1 , y=convolved image 14x14x3
    :return: dl_dw and dl_db loss derivatives wrt w and b
    dl_dw = dl_dy(from loss) * dy_dw(=x) = sigma k * sigma l * L_y * Xk+i,l+j
    '''
    H, W, C1 = np.shape(x)  # expecting 14x14x1
    h, w, C1, C2 = np.shape(w_conv)
    dl_dw, dl_db = np.zeros(np.shape(w_conv)), np.zeros(np.shape(b_conv))

    # we need to also consider those pads while convolving with the filter in forward pass
    padded_x = np.pad(x, ((1,1), (1,1), (0,0)), 'constant', constant_values=(0, 0))

    for out_c in range(C2): # easy peasy 1 dim >> maybe make it shape (3,1)??
        dl_db[out_c] = np.sum(dl_dy[:,:,out_c])

    for out_c in range(C2):
        for in_c in range(C1):
            for dw_h in range(h):
                for dw_w in range(w):
                    dl_dw[dw_h, dw_w, in_c, out_c] = np.sum(
                        np.multiply(dl_dy[:, :, out_c], padded_x[dw_h:dw_h+H, dw_w:dw_w+W, in_c])
                    )
    return dl_dw, dl_db

def pool2x2(x):
    '''
    :param x: H x W x C
    :return y: H/2 x W/2 x C (7x7x3)
    #>>> for each channel c, do max pool 2x2 with stride 2 <<<
    '''
    stride = 2
    max_pool_size = 2
    H, W, C = np.shape(x)
    y = np.zeros((int(H/max_pool_size),int(W/max_pool_size),C))

    for c in range(C):
        #print(x[:,:,c])
        for h in range(0, H, stride): # bc of the stride 2
            for w in range(0, W, stride):
                #print(x[h:h+max_pool_size, w:w+max_pool_size, c])
                y_h, y_w = int(h/max_pool_size), int(w/max_pool_size) # just make sure it won't be float!
                y[y_h,y_w,c] = np.max(x[h:h+max_pool_size, w:w+max_pool_size, c])
    return y

def pool2x2_backward(dl_dy, x, y):
    '''
    :return: we gotta upsample here!
    '''
    stride = 2
    max_pool_size = 2
    H, W, C = np.shape(x)
    dl_dx = np.zeros((H, W, C)) # should be the same size as x

    for c in range(C):
        for h in range(0, int(H/2)):  # bc of the stride 2
            for w in range(0, int(W/2)):
                # we gotta find the position here and put the max there!
                temp_x = x[h:h+max_pool_size, w:w+max_pool_size, c]
                max_pos = np.argmax(temp_x) # just for the position

                if max_pos == 0:
                    dl_dx[h * max_pool_size, w * max_pool_size, c] = dl_dy[h, w, c]
                elif max_pos == 1:
                    dl_dx[h * max_pool_size, w * max_pool_size + 1, c] = dl_dy[h, w, c]
                elif max_pos == 2:
                    dl_dx[h * max_pool_size + 1, w * max_pool_size, c] = dl_dy[h, w, c]
                elif max_pos == 3:
                    dl_dx[h * max_pool_size + 1, w * max_pool_size + 1, c] = dl_dy[h, w, c]
    return dl_dx


def flattening(x):
    '''
    :param x: HxWxC
    :return: y H*W*C  vectorized tensor! column major
    '''
    y = np.ndarray.flatten(x)
    y = y.reshape((len(y), 1)) # need this reshape for FC
    return y


def flattening_backward(dl_dy, x, y):
    '''
    :return: loss derivative wrt x
    '''
    shape = np.shape(x)
    dl_dx = np.reshape(dl_dy, shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    '''
    :param mini_batch_x, mini_batch_y:  cells from get_mini_batch
    :return: w 10x196
    :return: b 10x1
    Accuracy near 30%
    '''
    learning_rate = 0.01  # 1 set the learning rate (from ML class)
    decay_rate = 0.9 # 2 set the decay rate
    num_iter = 5000
    loss_list = []
    img_size, batch_size = np.shape(mini_batch_x[0])
    # 3 initialize weights with random gaussian noise (0,1) 10 outputs, each image 196 pixels
    # indicated in the algorithm that mean=0 and standard deviation=1
    w = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, IMAGE_SIZE))
    # >> what about b? np.zeros or gaussian again?
    # b = np.zeros((NUM_CLASSES, 1))
    b = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, 1))

    # train loop
    Loss = 0
    Kth_batch = 0  # 4
    for i in range(num_iter):  # 5
        #print(f'Iteration {i}/{num_iter} and loss is {Loss}')
        if ((i+1) % 1000) == 0:  # 6 at every 1000th iteration, lr=decay*lr
            learning_rate *= decay_rate
        dl_dw, dl_db = np.zeros((NUM_CLASSES, IMAGE_SIZE)), np.zeros((NUM_CLASSES, 1))  # 7
        Loss = 0
        for b in range(batch_size):  # 8 for each image x in Kth mini batch:
            # forward propogation
            x = mini_batch_x[Kth_batch][:, b]  # image from mini batch
            y = mini_batch_y[Kth_batch][:, b]  # ground truth
            y_tilde = fc(x.reshape(IMAGE_SIZE, 1), w, b)  # 9 label prediction(nx1)

            l, dl_dy = loss_euclidean(y_tilde.reshape(-1), y)  # 10 loss computation
            Loss += abs(l)

            # back propagation
            dl_dx_update, dl_dw_update, dl_db_update = fc_backward(dl_dy, x, w, b, y_tilde)  # 11 back-propagation
            dl_dw += dl_dw_update.reshape(NUM_CLASSES, IMAGE_SIZE)  # 12
            dl_db += dl_db_update.reshape(NUM_CLASSES, 1)

        loss_list.append(Loss)  # 1/M??
        Kth_batch += 1  # 14
        if Kth_batch >= len(mini_batch_x):
            Kth_batch = 0
        w -= (dl_dw / batch_size) * learning_rate  # 15
        b -= (dl_db / batch_size) * learning_rate

    visualize_loss(loss_list)
    return w, b


def train_slp(mini_batch_x, mini_batch_y):
    '''
    :param mini_batch_x, mini_batch_y:  cells from get_mini_batch
    :return: w 10x196
    :return: b 10x1
    '''
    learning_rate = 0.7  # 1 set the learning rate (from ML class)
    decay_rate = 0.9  # 2 set the decay rate
    num_iter = 8000
    loss_list = []
    img_size, batch_size = np.shape(mini_batch_x[0])
    # 3 initialize weights with random gaussian noise (0,1) 10 outputs, each image 196 pixels
    # indicated in the algorithm that mean=0 and standard deviation=1
    w = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, IMAGE_SIZE))
    b = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, 1))

    # train loop
    Loss = 0
    Kth_batch = 0  # 4
    for i in range(num_iter):  # 5
        #print(f'Iteration {i}/{num_iter} and loss is {Loss}')
        if ((i+1) % 1000) == 0:  # 6 at every 1000th iteration, lr=decay*lr
            learning_rate *= decay_rate
        dl_dw, dl_db = np.zeros((NUM_CLASSES, IMAGE_SIZE)), np.zeros((NUM_CLASSES, 1))  # 7
        Loss = 0
        for b in range(batch_size):  # 8 for each image x in Kth mini batch:
            # forward propagation
            x = mini_batch_x[Kth_batch][:, b]  # image from mini batch
            y = mini_batch_y[Kth_batch][:, b]  # ground truth
            y_tilde = fc(x.reshape(IMAGE_SIZE, 1), w, b)  # 9 label prediction(nx1)

            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)  # 10 loss computation
            Loss += abs(l)

            # back propagation
            dl_dx_update, dl_dw_update, dl_db_update = fc_backward(dl_dy, x, w, b, y_tilde)  # 11 back-propagation
            dl_dw += dl_dw_update.reshape(np.shape(dl_dw))  # 12
            dl_db += dl_db_update.reshape(np.shape(dl_db))

        loss_list.append(Loss)  # 1/M??
        Kth_batch += 1  # 14
        if Kth_batch >= len(mini_batch_x):
            Kth_batch = 0
        w -= (dl_dw / batch_size) * learning_rate  # 15
        b -= (dl_db / batch_size) * learning_rate

    visualize_loss(loss_list)
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    '''
    :param mini_batch_x:
    :param mini_batch_y:
    :return: w1 (30x196) b1 (30x1) w2 (10x30) b2 (10x1)
    '''
    learning_rate = 0.2 # 0.04, 0.7 84%  0.05,0.9 86%  0.05,0.98 20000 89%relu 0
    decay_rate = 0.9
    num_iter = 10000
    loss_list = []
    img_size, batch_size = np.shape(mini_batch_x[0])

    w1 = np.random.normal(loc=0, scale=1, size=(NUM_HID_LAYERS, IMAGE_SIZE))
    w2 = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, NUM_HID_LAYERS))
    b1 = np.random.normal(loc=0, scale=1, size=(NUM_HID_LAYERS, 1))
    b2 = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, 1))

    # train loop
    Loss = 0
    Kth_batch = 0  # 4
    for i in range(num_iter):  # 5
        print(f'Iteration {i}/{num_iter} and loss is {Loss}')
        if ((i+1) % 1000) == 0:  # 6 at every 1000th iteration, lr=decay*lr
            learning_rate *= decay_rate
        dl_dw1, dl_db1 = np.zeros((NUM_HID_LAYERS, IMAGE_SIZE)), np.zeros((NUM_HID_LAYERS, 1))  # 7
        dl_dw2, dl_db2 = np.zeros((NUM_CLASSES, NUM_HID_LAYERS)), np.zeros((NUM_CLASSES, 1))
        Loss = 0
        for b in range(batch_size):  # 8
            # forward propagation
            x = mini_batch_x[Kth_batch][:, b]  # image from mini batch
            y = mini_batch_y[Kth_batch][:, b]  # ground truth
            y_tilde1 = fc(x.reshape(IMAGE_SIZE, 1), w1, b1)  # 9 label prediction(nx1)
            y_tilde1_relu = relu(y_tilde1)
            y_tilde2 = fc(y_tilde1_relu.reshape(30, 1), w2, b2)

            l, dl_dy = loss_cross_entropy_softmax(y_tilde2, y)  # 10 loss computation
            Loss += abs(l)

            # back propagation
            dl_dx_update2, dl_dw_update2, dl_db_update2 = fc_backward(dl_dy, y_tilde1_relu, w2, b2, y_tilde2)
            relu_back = relu_backward(dl_dx_update2, y_tilde1, y_tilde1_relu)
            dl_dx_update1, dl_dw_update1, dl_db_update1 = fc_backward(relu_back, x, w1, b1, y_tilde1)


            dl_dw1 += dl_dw_update1.reshape(np.shape(dl_dw1))  # 12
            dl_db1 += dl_db_update1.reshape(np.shape(dl_db1))
            dl_dw2 += dl_dw_update2.reshape(np.shape(dl_dw2))  # 12
            dl_db2 += dl_db_update2.reshape(np.shape(dl_db2))
            
        # print(abs(l))
        loss_list.append(Loss)
        Kth_batch += 1  # 14
        if Kth_batch >= len(mini_batch_x):
            Kth_batch = 0
        w1 -= (dl_dw1 / batch_size) * learning_rate  # 15
        b1 -= (dl_db1 / batch_size) * learning_rate
        w2 -= (dl_dw2 / batch_size) * learning_rate
        b2 -= (dl_db2 / batch_size) * learning_rate

    visualize_loss(loss_list)
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    '''
    input 1ch -> 3x3 conv -> 3ch output (stride 1) -> ReLu -> max pooling 2x2 (stride 2) -> flattening -> FC -> softmax
    :param mini_batch_x:
    :param mini_batch_y:
    :return: w_conv 3x3x1x3 , b_conv 3, w_fc 10x147, b_fc 10x1
    ''' # 0.002+0.7 8000-> 45%,  0.002+0.5 5000-> 35%  0.03+0.9 5000 86%, 0.03+0.98 20,000-> 88.9%
    # overnight 0.02+0.9 20k *0.1 to Ws = 88%, 0.38+0.9 20k -> 89.5%, 0.35+0.7 20k 0.1w->
    learning_rate = 0.04
    decay_rate = 0.95
    num_iter = 20000
    loss_list = []
    img_size, batch_size = np.shape(mini_batch_x[0])

    w_conv = np.random.normal(loc=0, scale=1, size=(CONV_SIZE,CONV_SIZE,1,CONV_SIZE))
    b_conv = np.random.normal(loc=0, scale=1, size=(CONV_SIZE))
    w_fc = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, FLAT_LAYER))
    b_fc = np.random.normal(loc=0, scale=1, size=(NUM_CLASSES, 1))
    #b_conv = np.zeros((CONV_SIZE))
    #b_fc = np.zeros((NUM_CLASSES, 1))
    # train loop
    Loss = 0
    Kth_batch = 0  # 4
    for i in range(num_iter):  # 5
        print(f'Iteration {i}/{num_iter} and loss is {Loss}')
        #if i==100: learning_rate = 0.1
        #if i==1000: learning_rate = 0.03
        if ((i+1) % 1000) == 0:  # 6 playing with this number hoping to increase the accuracy
            learning_rate *= decay_rate
        #if ((i+1) % 2000) == 0:
        #    learning_rate *= decay_rate * 0.5
        dl_dw_conv, dl_db_conv = np.zeros((CONV_SIZE,CONV_SIZE,1,CONV_SIZE)), np.zeros((CONV_SIZE))  # 7
        dl_dw_fc, dl_db_fc = np.zeros((NUM_CLASSES, FLAT_LAYER)), np.zeros((NUM_CLASSES, 1))
        Loss = 0
        for b in range(batch_size):  # 8
            # forward propagation
            x = mini_batch_x[Kth_batch][:, b].reshape((14, 14, 1), order='F')# Fortran-like index order! necessary
            y = mini_batch_y[Kth_batch][:, b]  # ground truth
            #x = np.reshape(x, (14,14,1))

            #inpiut -> convolve 3x3x1x3 -> ReLu -> max pool -> flattening -> FC -> softmax
            x_conv = conv(x, w_conv, b_conv)
            x_conv_relu = relu(x_conv)
            x_conv_relu_pool = pool2x2(x_conv_relu)
            x_conv_relu_pool_flat = flattening(x_conv_relu_pool)
            x_conv_relu_pool_flat_fc = fc(x_conv_relu_pool_flat, w_fc, b_fc) # check if i have to reshape it (xxx,1)

            l, dl_dy = loss_cross_entropy_softmax(x_conv_relu_pool_flat_fc, y)
            Loss += abs(l)

            # back propagation
            dl_dx_update, dl_dw_fc_update, dl_db_fc_update = fc_backward(dl_dy, x_conv_relu_pool_flat, w_fc, b_fc, x_conv_relu_pool_flat_fc)
            flat_back = flattening_backward(dl_dx_update, x_conv_relu_pool, x_conv_relu_pool_flat)
            pool_back = pool2x2_backward(flat_back, x_conv_relu, x_conv_relu_pool)
            relu_back = relu_backward(pool_back, x_conv, x_conv_relu)
            dl_dw_conv_update, dl_db_conv_update = conv_backward(relu_back, x, w_conv, b_conv, x_conv)

            dl_dw_conv += dl_dw_conv_update.reshape(np.shape(dl_dw_conv))  # 12
            dl_db_conv += dl_db_conv_update.reshape(np.shape(dl_db_conv))
            dl_dw_fc += dl_dw_fc_update.reshape(np.shape(dl_dw_fc))  # 12
            dl_db_fc += dl_db_fc_update.reshape(np.shape(dl_db_fc))

        # print(abs(l))
        loss_list.append(Loss)
        Kth_batch += 1  # 14
        if Kth_batch >= len(mini_batch_x):
            Kth_batch = 0
            #np.random.shuffle(mini_batch_x)
        w_conv -= (dl_dw_conv / batch_size) * learning_rate # 15
        b_conv -= (dl_db_conv / batch_size) * learning_rate
        w_fc -= (dl_dw_fc / batch_size) * learning_rate
        b_fc -= (dl_db_fc / batch_size) * learning_rate

    visualize_loss(loss_list)
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



