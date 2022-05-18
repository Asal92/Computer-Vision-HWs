import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def get_differential_filter():
    # To do
    # filter_x/y are 3x3 filters that differentiate along x and y directions respectively
    filter_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float)
    filter_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)

    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    # im is mxn grayscale image
    # filter is kxk matrix
    # add padding pixels to get the same size filtered image cameraman=256x256
    im_filtered = np.zeros((np.shape(im)))
    im = np.pad(im, (1,1), 'constant', constant_values=(0,0)) # image becomes 258x258 after padding
    m,n = np.shape(im)

    for i in range(0,m-2):
        for j in range(n-2):
            im_filtered[i][j] = np.sum(np.multiply(im[i:i+3,j:j+3], filter))

    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do >>> elementwise  no unsigned angle!!!
    # im_dx,im_dy = differential images
    # grad_mag = gradient magnitude, grad_angle=orientation of gradient [0,pi) unsigned angle ( 0 = 0 + pi )
    # they should be the same size as the main image
    m,n = np.shape(im_dx)
    grad_mag = np.zeros((m,n))
    grad_angle = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            grad_mag[i, j] = np.sqrt((im_dx[i, j] ** 2) + (im_dy[i, j] ** 2))
            grad_angle[i, j] = np.arctan2(im_dy[i, j], im_dx[i, j])
            if grad_angle[i, j] < 0:
                grad_angle[i, j] += np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    # m,n size of the gradient array  M,N Number of the cells
    m,n = np.shape(grad_mag)
    M = int(m/cell_size)
    N = int(n/cell_size)

    ori_histo = np.zeros((M,N,6))
    # convert gradinet angle to degrees to compute the histogram
    grad_degree = np.degrees(grad_angle)

    # 6 bins in total
    # θ1 = [165◦, 180◦) ∪ [0◦, 15◦), θ2 = [15◦, 45◦), θ3 = [45◦, 75◦), θ4 = [75◦, 105◦), θ5 = [105◦, 135◦), and θ6 = [135◦, 165◦)
    for i in range(M):
        for j in range(N):
            for k in range(cell_size):
                for l in range(cell_size):
                    if 165 <= grad_degree[i*cell_size + k, j*cell_size + l] < 180 or 0 <= grad_degree[i*cell_size + k, j*cell_size + l] < 15:
                        ori_histo[i, j, 0] += grad_mag[i*cell_size + k][j*cell_size + l]
                    elif 15 <= grad_degree[i*cell_size + k, j*cell_size + l] < 45:
                        ori_histo[i, j, 1] += grad_mag[i*cell_size + k][j*cell_size + l]
                    elif 45 <= grad_degree[i*cell_size + k, j*cell_size + l] < 75:
                        ori_histo[i, j, 2] += grad_mag[i*cell_size + k][j*cell_size + l]
                    elif 75 <= grad_degree[i*cell_size + k, j*cell_size + l] < 105:
                        ori_histo[i, j, 3] += grad_mag[i*cell_size + k][j*cell_size + l]
                    elif 105 <= grad_degree[i*cell_size + k, j*cell_size + l] < 135:
                        ori_histo[i, j, 4] += grad_mag[i*cell_size + k][j*cell_size + l]
                    elif 135 <= grad_degree[i*cell_size + k, j*cell_size + l] < 165:
                        ori_histo[i, j, 5] += grad_mag[i*cell_size + k][j*cell_size + l]
    #print(ori_histo)
    # ori_histo size will be:  M x N x 6(# of bins)
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    # 1- Normalizing ori_histo (L2 normalization) hi_hat = hi / sqrt(sigma hi2 + e2)
    # 2- grouping and concatenating 2 cells horizontally and vertically
    e = 0.001
    M,N,bin = np.shape(ori_histo) # we know bin is 6 here!
    # normalized histogram ignores last row and column
    ori_histo_normalized = np.zeros((M-(block_size-1),N-(block_size-1),(6*block_size*block_size)))

    for i in range(M-(block_size-1)): # avoid edges
        for j in range(N-(block_size-1)):
            # hi will be array of 24 elements after concatenating cells
            hi = np.reshape(ori_histo[i:i + block_size, j:j + block_size], (6 * block_size * block_size,))
            #print(np.shape(hi))
            sigma = np.sqrt((np.sum(hi ** 2)) + (e ** 2))
            ori_histo_normalized[i,j] = hi / sigma
            #for k in range(6*block_size*block_size):
                #ori_histo_normalized[i][j][k] = hi[k] / sigma

    #print(np.shape(ori_histo_normalized))
    # ori_histo_normalized is 31x31x24 array for cameraman
    return ori_histo_normalized


def extract_hog(im):
    # 1.Convert the gray-scale image to float format and normalize to range [0, 1].
    im = im.astype('float') / 255.0 # Division by 255 makes the normalization 0-1
    #print(im)
    # To do
    # 2. Get differential images using get_differential_filter and filter_image
    filter_x, filter_y = get_differential_filter()
    im_filtered_x = filter_image(im, filter_x)
    im_filtered_y = filter_image(im, filter_y)

    # 3.Compute the gradients using get_gradient
    grad_mag, grad_angle = get_gradient(im_filtered_x, im_filtered_y)

    # 4. Build the histogram of oriented gradients for all cells using build_histogram
    ori_histo = build_histogram(grad_mag, grad_angle, 8) # it's stated in HW that cell size is usually 8

    # 5. Build the descriptor of all blocks with normalization using get_block_descriptor
    descriptor = get_block_descriptor(ori_histo, 2)
    hog = descriptor.reshape((-1))
    # visualize to verify
    # I had to move this to main, because it was trying to visualize every bounding box for target and was taking so long
    #visualize_hog(im, hog, 8, 2)

    # 6.Return a long vector (hog) by concatenating all block descriptors.
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()



def face_recognition(I_target, I_template):
    # NCC = a.b/||a||||b||
    # ai = ai - a~
    # 2 suppressions are happening for face recognition: 1-with thresholding using NCC 2-NMS with IoU>0.5

    # hog for template
    b = extract_hog(I_template)
    b -= np.mean(b) # Normalizing template's hog for NCC
    b_norm = np.linalg.norm(b) # for ||b||

    # getting the sizes of template & target & the difference for # of bounding boxes
    M_template, N_template = np.shape(I_template)
    M_target, N_target = np.shape(I_target)
    M_diff = M_target - M_template
    N_diff = N_target - N_template

    # set of bounding boxes and boudning box itself that contains (xi,yi,si) information of the box
    bb_set = np.empty((0,3))
    bounding_boxes = np.empty((0,3))
    # After multiple attempts, I found 0.49 to be a good threshold
    thresh = 0.49
    # this takes very very long!!!! please be patient
    for i in range(0,M_diff):
        for j in range(0,N_diff):
            # getting hog from target with size of template and normalizing it
            a_box = extract_hog(I_target[i:i+M_template, j:j+N_template])
            a_box -= np.mean(a_box)
            a_norm = np.linalg.norm(a_box) # this if for getting ||a||
            # NCC score
            s = np.sum(a_box*b)/ (a_norm*b_norm)
            # No.1 suppression. Adding only boxes that have thresh>0.42
            if s >= thresh:
                # I had to swap i & j to use N for x coordinate and M for y coordinate
                bb_set = np.vstack((bb_set,(j,i,s))) # this is sooo confusing!!!

    # No.2 Non-Maximum Suppression Algorithm for IoU>0.5
    while len(bb_set) > 0:
        # 1. find the BB of maximum score from BB set
        max_score = -np.inf
        index = 0
        for i, box in enumerate(bb_set):
            if box[2]> max_score:
                max_bb = bb_set[i]
                max_score = box[2]
                index = i
        #print(max_score)
        # adding box with max score to bounding box set and delete from original bb set
        bounding_boxes = np.vstack((bounding_boxes, max_bb))
        bb_set = np.delete(bb_set, index, axis=0)

        # 2. Suppress all BBs in BB set that have IoU greater than 0.5
        suppress_list = []
        for i, box in enumerate(bb_set):
            # x-y coordinates of the intersection top-left and bottom-right corners
            x_left = max(max_bb[0], box[0])
            y_top = max(max_bb[1], box[1])
            x_right = min(max_bb[0]+N_template, box[0]+N_template)
            y_bottom = min(max_bb[1]+M_template, box[1]+M_template)
            # area of intersection
            intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            # finally IoU using intersection and template size
            IoU = intersection / float((M_template * N_template) * 2 - intersection)
            if IoU > 0.5:  # NMS Suppression
                suppress_list.append(i)
        # delete boxes with IoU>0.5 from original bb set
        bb_set = np.delete(bb_set,suppress_list, axis=0)
        if len(bb_set) == 0:
            break

    return bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    # By running this code, first you will see hog visualization of the camerman
    # Close the camerman and wait until the face recognition visualization pops up.
    # It will take quite long time (at least 5 minutes) to detect faces, please be patient :) 

    
    # 256 x 256 image
    im = cv2.imread('cameraman.tif', 0)
    im_float = im.astype('float') / 255.0
    hog = extract_hog(im)
    visualize_hog(im_float, hog, 8, 2)

    # MxN image
    I_target= cv2.imread('target.png', 0)
    # bellow code is for target hog visualization...
    #Im_target_float = I_target.astype('float') / 255.0
    #hog_target = extract_hog(I_target)
    #visualize_hog(Im_target_float, hog_target, 8, 2)

    # mxn  face template
    I_template = cv2.imread('template.png', 0)
    Im_template_float = I_template.astype('float') / 255.0
    # bellow code is for template hog visualization...
    #hog_template = extract_hog(I_template)
    #visualize_hog(Im_template_float, hog_template, 8, 2)


    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, Im_template_float.shape[0])
    #this is visualization code.

