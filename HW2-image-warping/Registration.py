import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import matplotlib
matplotlib.use('TkAgg')


NEIGHBOR_THRESHOLD = 0.7
ICIA_ITER = 200
DELTA_P_NORM_THRESHOLD = 0.01
RANSAC_ITER = 10000
RANSAC_THRESH = 10


'''1. Find SIFT keypoints 
2. find matchign points using KNN
3. filter matching points with ratio d1/d2 < 0.7
4. RANSAC algo in a loop affine_iter:
    find Affine transformation matrix using matching points
    apply affine to template points
    compute euclidean distance between transformed template and target
    if distance is less than a threshold: count inliers
    return the affine matrix with max number of inliers
5. Warp the template with the Affine matrix from RANSAC and see the error image
6. Align template and target from best affine model using "Inverse Compositional" Algorithm
'''

def find_match(img1, img2):
    # To do

    x1 = np.empty((0,2))
    x2 = np.empty((0, 2))
    x3 = np.empty((0, 2))
    x4 = np.empty((0, 2))
    x1_match = np.empty((0, 2))
    x2_match = np.empty((0, 2))

    # 1. construct a SIFT object >>>> visualize SIFT
    #sift = cv2.xfeatures2d.SIFT_create() >> This gives me an error! even though its suggested by library
    sift = cv2.SIFT_create()

    # 2. find keypoints and descriptors(mxnx128 ndarray) in template and target >>> keypoint 128???
    template_kp, template_des = sift.detectAndCompute(img1, None)
    target_kp, target_des = sift.detectAndCompute(img2, None)
    '''# Visualizing SIFT keypoints
    img11 = cv2.drawKeypoints(img1, template_kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('template_keypoints.jpg', img11)
    img22 = cv2.drawKeypoints(img2, target_kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('target_keypoints.jpg', img22)'''

    # converting keypoints classes to x-y coordinates for both template and target
    template_xy = [p.pt for p in template_kp]
    target_xy   = [p.pt for p in target_kp]
    #print(len(template_kp), len(target_kp))

    # 3. find matches using nearest neighbor with ratio test >>> 2 sides with ratio test? do we need to calculate norm??
    # dist,ind = num of matching points x 2  . returned indices are actually input data indexes (target_des)
    # ind = 2 nearest points in target to the keypoint i of template, first elementind[0] is the closest

    # from template to taregt
    neigh1 = NearestNeighbors(n_neighbors=2)
    neigh1.fit(target_des)
    dist1, ind1 = neigh1.kneighbors(template_des) # returns distance and indices of nearest points
    for i in range(len(ind1)):
        d1,d2 = dist1[i]
        if d1/d2 < NEIGHBOR_THRESHOLD:
            x1 = np.vstack((x1, template_xy[i]))
            x2 = np.vstack((x2, target_xy[ind1[i,0]]))

    # from target to template
    neigh2 = NearestNeighbors(n_neighbors=2)
    neigh2.fit(template_des)
    dist2, ind2 = neigh2.kneighbors(target_des)
    for i in range(len(ind2)):
        d1,d2 = dist2[i]
        if d1/d2 < NEIGHBOR_THRESHOLD:
            x3 = np.vstack((x3, template_xy[ind2[i, 0]]))
            x4 = np.vstack((x4, target_xy[i]))

    # 4. Bi-directional Consistency Check
    for i in range(len(x1)):
        if x1[i] in x3 and x2[i] in x4:
            x1_match = np.vstack((x1_match, x1[i]))
            x2_match = np.vstack((x2_match, x2[i]))

    print(f'matching points lenghts are {len(x1_match)} and {len(x2_match)}')
    print("matching points after nearest neighbors:")
    print(x1_match, x2_match)


    return x1_match, x2_match

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    # A should be 3x3 matrix (reshape 6x1)  3 random data points , 6 unknowns
    best_affine_model = np.zeros((3,3))
    max_num_inliers = -np.inf

    for i in range(ransac_iter):
        # 1. Random Sampling - pick 3 random key points of both x1 and x2
        # replace=False makes sure there aren't any duplicates for random choices
        u = np.random.choice(len(x1), size=3, replace=False)
        #v = np.random.choice(len(x2), size=3, replace=False)
        u1,u2,u3 = x1[u[0]],x1[u[1]],x1[u[2]]
        v1,v2,v3 = x2[u[0]],x2[u[1]],x2[u[2]] # should be the same points

        A = np.array([[u1[0], u1[1], 1, 0, 0, 0],
                      [0, 0, 0, u1[0], u1[1], 1],
                      [u2[0], u2[1], 1, 0, 0, 0],
                      [0, 0, 0, u2[0], u2[1], 1],
                      [u3[0], u3[1], 1, 0, 0, 0],
                      [0, 0, 0, u3[0], u3[1], 1]])

        b = np.array([[v1[0],v1[1],v2[0], v2[1], v3[0], v3[1]]]).T

        # x = (A.T * A ).inverse * A.T * b - try np.linalg.solve ?!!
        #x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)), A.T),b)
        try:
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)), A.T),b)
        except np.linalg.LinAlgError: # added this since I was getting the error!!
            continue

        #6x1 array -> reshape to 3x3
        affine_matrix = np.append(x,[0,0,1]) # Adding third row of Affine matrix
        affine_matrix = np.reshape(affine_matrix,(3,3))
        #print(np.shape(affine))
        #print(affine)

        # 2. Building a model - transform template with Affine, compute euclidean distance between template and target matching points,
        # finally check with threshold count inliers
        num_inliers = 0
        for i in range(len(x1)):
            x1_point = np.append(x1[i],1)
            x2_point = np.append(x2[i], 1)
            x1_point_affine_transformed = np.matmul(affine_matrix, x1_point)
            #print(np.shape(x1_point_affine_transformed))
            euc_dist = np.sqrt(np.sum(np.power((x1_point_affine_transformed - x2_point),2)))
            #print(euc_dist)

            # 3,4. Thresholding and inlier counting
            if euc_dist < ransac_thr:
                num_inliers += 1

        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_affine_model = affine_matrix

    print(f'max num of inliers is: {max_num_inliers}')
    print("best Affine transform after RANSAC:")
    print(best_affine_model)

    return best_affine_model # replaced A to avoid confusion with Affine model


def warp_image(img, A, output_size):
    # To do
    # Used slide's method instead of interpolate methopd in scipy library
    m,n = output_size
    img_warped = np.zeros((m,n))

    '''
    # I made an attempt to run with interpolation but warped image won't be correct, I don't understand why.
    # doing x,y alone in interpolation didn't wok!!
    # I tried to apply Affine matrix to my x,y somehow, I tried to play with x,y inputs
    # subtraction below made it to work but I have no explanation for why it's this way!
    # since I am not comfortable with this code, I will comment it out and use Prof. method instead
    
    x = np.linspace(0, img.shape[0]-1, img.shape[0]) # 1-D array of size shape[0]
    y = np.linspace(0, img.shape[1]-1, img.shape[1]) # 1-D array of size shape[1]
    interp = interpolate.RectBivariateSpline(x - (A[1,2]), y - (A[0,2]), img)

    for i in range(m):
        for j in range(n):
            img_warped[i,j] = interp.ev(i,j)'''

    for i in range(m):
        for j in range(n):
            x2 = np.array([[j,i,1]])
            x1 = np.matmul(A,x2.T)
            img_warped[i,j] = img[int(x1[1]),int(x1[0])]

    '''# Visualizing the error
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)
    I_error = abs(template - img_warped)
    plt.imshow(I_error,  cmap='coolwarm')
    plt.show()'''

    return img_warped


def align_image(template, target, A):
    # To do

    template_m, template_n = np.shape(template)
    Hessian = np.zeros((6, 6))
    delta_p = np.ones((6, 1))
    steepest_descent_images = np.zeros((template_m, template_n, 6))

    # Gradient of template - remember top have delta_I 1x2 matrix
    '''
    I found out that with sobel filter, aligning takes around 500 iterations
    but if with just the derivative filter I can make good results for about 200 iterations
    '''
    #filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], int)
    #filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], int)
    filter_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], int)
    filter_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], int)



    grad_x, grad_y = np.zeros((np.shape(template))), np.zeros((np.shape(template)))
    im = np.pad(template, (1, 1), 'constant', constant_values=(0, 0))  # image becomes 258x258 after padding
    m, n = np.shape(im)

    for i in range(0, m - 2):
        for j in range(n - 2):
            grad_x[i][j] = np.sum(np.multiply(im[i:i + 3, j:j + 3], filter_x))
            grad_y[i][j] = np.sum(np.multiply(im[i:i + 3, j:j + 3], filter_y))

    # 2. compute delta I
    #delta_I = np.zeros((m, n, 2))

    # 3,4. Compute Steepest decent images.  gradient 1x2 * jacobian 2x6 (rememver to swap x and y for jacobian!)
    for m in range(template_m):
        for n in range(template_n):
            delta_I = np.array([grad_x[m, n], grad_y[m, n]])
            jacobian = np.array([[n, m, 1, 0, 0, 0],
                                 [0, 0, 0, n, m, 1]])
            steepest_descent_images[m, n] = np.matmul(delta_I, jacobian)


    # 5. Compute Hessian matrix now
    for m in range(template_m):
        for n in range(template_n):
            temp = np.matmul(steepest_descent_images[m, n].reshape(6, 1), steepest_descent_images[m, n].reshape(1, 6))
            Hessian += temp

    Hessian_inv = np.linalg.inv(Hessian)

    # 6. Refining warp function (here affine transform)
    errors = []
    iter = 0
    last_error = np.inf

    while True:
        iter += 1

        # 7. Warp target to template domain
        target_warp = warp_image(target, A, np.shape(template))

        # 8. Compute error image
        I_error = target_warp - template

        # 9. Compute F
        F = np.zeros((6, 1))
        for m in range(template_m):
            for n in range(template_n):
                F += ((steepest_descent_images[m, n]).T * I_error[m, n]).reshape(6, 1)

        # 10. Compute delta p
        delta_p = np.matmul(Hessian_inv, F)
        delta_p_norm = np.linalg.norm(delta_p)

        Wp = np.append(delta_p, [0, 0, 1])
        Wp = np.reshape(Wp, (3, 3))
        Wp[0, 0] += 1
        Wp[1, 1] += 1

        # 11. Update Warp
        A = np.matmul(A, np.linalg.inv(Wp))
        # print(A)

        error_norm = np.linalg.norm(I_error)
        #error_average = np.sum(np.abs(I_error)) / (template.size * 255)
        error_average = error_norm / template.size # per TA's suggestion
        errors = np.append(errors, error_average)
        # print(iter, (np.sum(np.abs(I_error)) / template.size), np.linalg.norm(delta_p))
        print(f'iter#: {iter}, error norm: {error_norm}, error_average: {error_average*255}, delta_p norm: {delta_p_norm}')

        # error will start decreasing but after some iterations, the error starts increasing instead of decreasing!!
        # I will add extra check that if error starts increasing, it should break out of the loop.
        # this has been checked with TA
        if error_norm < last_error:
            last_error = np.linalg.norm(I_error)
        elif error_norm > last_error + 20: # 20 or more maybe?
            break

        if iter > ICIA_ITER or delta_p_norm < DELTA_P_NORM_THRESHOLD:
            break

    ''' one image visualization only:
    boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                      [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
    plt.subplot(221)
    plt.imshow(target, cmap='gray')
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.show()'''

    A_refined = A
    return A_refined, errors


def track_multi_frames(template, img_list):
    # To do
    A_list = [] # set of Affine transforms from template to each target
    # find matches first
    x1, x2 = find_match(template, img_list[0])
    # get the initial Affine
    A = align_image_using_feature(x1, x2, RANSAC_THRESH, RANSAC_ITER)

    # now align target images and append the new Affine to list, then update (warp) template
    for target in img_list:
        A, errors = align_image(template, target, A)
        template = warp_image(target, A, template.shape)
        A_list.append(A)

    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)
    ransac_thr, ransac_iter = 10, 10000
    A = align_image_using_feature(x1, x2, RANSAC_THRESH, RANSAC_ITER)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


