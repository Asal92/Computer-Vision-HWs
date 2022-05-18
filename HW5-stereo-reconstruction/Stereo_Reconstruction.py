import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')


NEIGHBOR_THRESHOLD = 0.7
RANSAC_ITER = 500
RANSAC_THRESH = 0.05

def find_match(img1, img2):
    '''
    1.extract keypoints 2.match using nearest neighbors 3.filter matches using ratio test and bidirectional consistency
    :param img1, img2: two input gray-scale images uint8 format
    :return: nx2 matrices, each row is (x,y) coordinates
    '''

    x1 = np.empty((0, 2))
    x2 = np.empty((0, 2))
    x3 = np.empty((0, 2))
    x4 = np.empty((0, 2))
    x1_match = np.empty((0, 2))
    x2_match = np.empty((0, 2))

    # 1. construct sift and extracting keypoints
    sift = cv2.SIFT_create()
    template_kp, template_des = sift.detectAndCompute(img1, None)
    target_kp, target_des = sift.detectAndCompute(img2, None)
    template_xy = [p.pt for p in template_kp]
    target_xy = [p.pt for p in target_kp]

    # 2. matching using nearest neighbor

    # from template to taregt
    neigh1 = NearestNeighbors(n_neighbors=2)
    neigh1.fit(target_des)
    dist1, ind1 = neigh1.kneighbors(template_des)  # returns distance and indices of nearest points
    for i in range(len(ind1)):
        d1, d2 = dist1[i]
        if d1 / d2 < NEIGHBOR_THRESHOLD:
            x1 = np.vstack((x1, template_xy[i]))
            x2 = np.vstack((x2, target_xy[ind1[i, 0]]))

    # from target to template
    neigh2 = NearestNeighbors(n_neighbors=2)
    neigh2.fit(template_des)
    dist2, ind2 = neigh2.kneighbors(target_des)
    for i in range(len(ind2)):
        d1, d2 = dist2[i]
        if d1 / d2 < NEIGHBOR_THRESHOLD:
            x3 = np.vstack((x3, template_xy[ind2[i, 0]]))
            x4 = np.vstack((x4, target_xy[i]))

    # 3. Bi-directional Consistency Check
    for i in range(len(x1)):
        if x1[i] in x3 and x2[i] in x4:
            x1_match = np.vstack((x1_match, x1[i]))
            x2_match = np.vstack((x2_match, x2[i]))

    print(f'matching points lenghts are {len(x1_match)} and {len(x2_match)}')
    #print("matching points after nearest neighbors:")
    #print(x1_match, x2_match)
    pts1, pts2 = x1_match, x2_match
    return pts1, pts2


def compute_F(pts1, pts2):
    '''
    F is computed by 8point algo with RANSAC. rank of matrix should be 2!!
    SVD cleanup should be applied
    :param pts1, pts2: nx2 matrices from sift
    :return: F is 3x3 matrix
    '''

    max_num_inliers = -np.inf
    min_loss = np.inf
    for i in range(RANSAC_ITER):
        # 1. random sampling and forming the A matrix
        # replace=False makes sure there aren't any duplicates for random choices
        rand_pts = np.random.choice(len(pts1), size=8, replace=False)

        A = np.ones((8,9)) # 9 F parameters, but I need 8 points
        for n in range(8):
            A[n, 0] = pts1[rand_pts[n]][0] * pts2[rand_pts[n]][0]
            A[n, 1] = pts1[rand_pts[n]][1] * pts2[rand_pts[n]][0]
            A[n, 2] = pts2[rand_pts[n]][0]
            A[n, 3] = pts1[rand_pts[n]][0] * pts2[rand_pts[n]][1]
            A[n, 4] = pts1[rand_pts[n]][1] * pts2[rand_pts[n]][1]
            A[n, 5] = pts2[rand_pts[n]][1]
            A[n, 6] = pts1[rand_pts[n]][0]
            A[n, 7] = pts1[rand_pts[n]][1]

        # 2. estimate the F fundamental matrix
        # F should be nullspace(A)
        F_estimate = null_space(A)
        # apprently it gives more than one option sometimes!! maybe pick the first one only?
        F_estimate = F_estimate[:, 0]
        F_estimate = np.reshape(F_estimate, (3,3))

        # 3. svd cleanup. U D V.T
        u, d, v_t = np.linalg.svd(F_estimate) # get U(3x3), D(3,) V.T(3x3) matrices
        d[-1] = 0 # make it rank 2
        d = np.diag(d)
        F_svd = np.matmul(np.matmul(u,d),v_t)

        # building a model by counting the inliers
        '''num_inliers = 0
        err_list=[]
        for m in range(len(pts1)):
            ui = [pts1[m][0], pts1[m][1], 1]
            vi = [pts2[m][0], pts2[m][1], 1]
            err = np.matmul(np.matmul(vi, F_svd), ui) # vT.F.u = 0
            err_list.append(err)
            if err < RANSAC_THRESH:
                num_inliers += 1
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_F = F_svd'''
        err_list=[] # rather than counting inliers, checking which has the smallet error
        for m in range(len(pts1)):
            ui = [pts1[m][0], pts1[m][1], 1]
            vi = [pts2[m][0], pts2[m][1], 1]
            err = np.matmul(np.matmul(vi, F_svd), ui)  # vT.F.u = 0
            err_list.append(err**2) # power of 2 to avoid negatives
        loss = np.sum(err_list)
        if loss < min_loss:
            min_loss = loss
            best_F = F_svd

    print('error list:', err_list)
    print(f'max num of inliers is: {max_num_inliers}')
    print("best fundamental matrix after RANSAC:\n", best_F)
    F = best_F
    return F


def triangulation(P1, P2, pts1, pts2):
    '''
    :param P1, P2: camera projection matrices
    :param pts1, pts2: matching points from sift (320x2)
    :return: pts3D nx3 (320x3) - each row is 3D reconstructed point
    '''
    pts3D = np.empty((0,3))
    for i in range(len(pts1)):
        # two 3D vectors are in parallel
        u = [pts1[i][0], pts1[i][1], 1]
        v = [pts2[i][0], pts2[i][1], 1]
        # skew-symmetric matrix [a b c].T = [0 -c b][c 0 -a][-b a 0]
        u_skew = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        # [u 1]P
        u_skew_p = np.matmul(u_skew, P1)
        v_skew_p = np.matmul(v_skew, P2)
        # now we have to combine them in one 4x4 matrix - getting rid of third elements ??
        A = np.vstack((u_skew_p[:2], v_skew_p[:2]))
        # A[pts3D 1] = 0 calling it X to not confuse it with actuall pts3D
        X = null_space(A, rcond=0.1)
        X = X[:, 0] # sometimes it gives more than 1 null space
        X = X / X[3] # [X 1] the last element should be 1
        # pts3D has to be nx3 but X dim of 4!!
        pts3D = np.vstack((pts3D, X[:3]))
    print('the size of the pts3D:', np.shape(pts3D))
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    '''
    :param Rs, Cs: rotation and camera centers matrices
    :param pts3Ds: reconstructed points from triangulation
    :return:  best rotation, best camera centers, best 3D reconstructions
    '''
    best_nValid = -np.inf
    best_r, best_c, best_pts = 0,0,0
    for i in range(len(Rs)): # length of all three elements should be 4!
        # R 3x3, C 3x1 which should be reshaped, pts3D nx3
        r = Rs[i]
        r3 = r[2]
        c = Cs[i]
        c = c.reshape(-1) # this is so I can do X-C
        pts = pts3Ds[i]

        nValid = 0
        for x in pts: # pt 1x3
            # r3.T (X-C)>0  Cheirality condition for counting points in front of the camera
            cheirality = np.dot((x-c), r3)
            if cheirality > 0:
                nValid +=1
        if nValid > best_nValid:
            best_nValid = nValid
            best_r = r
            best_c = c
            best_pts = pts
    print("found the best camera pose!")
    R = best_r
    C = best_c.reshape((3,1))
    pts3D = best_pts
    return R, C, pts3D


def compute_rectification(K, R, C):
    '''
    :param K: intrinsic parameter
    :param R,C: rotation and center of the camera
    :return H1, H2: homographies the rectify left and right images
    H = K Rrect R.T K-1
    '''
    c = C.reshape(-1)
    #alignment between X axis and basleine
    r_x = c / np.linalg.norm(c) # 3x1
    r_z_t = np.array([0,0,1])
    r_z_d = ( r_z_t - (np.dot(r_z_t, r_x)) * r_x) # 3x1
    r_z = r_z_d / np.linalg.norm(r_z_d)
    r_y = np.cross(r_z, r_x) # 3x1

    R_rect = np.asarray([r_x, r_y, r_z]) # 3x3

    # H-BOB = K Rrect K-1  , H-Alice = K Rrect R.T K-1
    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    print("homogrophies are calculated!")
    return H1, H2


def dense_match(img1, img2):
    '''
    :param img1, img2: gray scale rectified images
    :return: disparity map H x W , image height and width
    '''
    H, W = np.shape(img1)
    disparity = np.ones((H, W))
    # computing SIFT descriptor for both images
    sift = cv2.SIFT_create()
    kp1, kp2=[], []
    for h in range(H): # >>>> x=height- row in image
        for w in range(W): # wirte up says every pixel!!
            kp1.append(cv2.KeyPoint(x=w, y=h, size=3))# swapping x,y!!!
            kp2.append(cv2.KeyPoint(x=w, y=h, size=10))
    img1_kp, img1_des = sift.compute(img1, kp1)
    img2_kp, img2_des = sift.compute(img2, kp2)
    img1_descriptor = np.reshape(img1_des, (H, W, 128))
    img2_descriptor = np.reshape(img2_des, (H, W, 128))
    print("dense SIFT descriptors are computed.")
    # d = argmin||d1u - d2u+(i,0)||**2
    for h in range(H):
        for w in range(W):
            d_list = []
            d1u = img1_descriptor[h,w] # SIFT at u for left image
            # d2u+(i,0) i=0,1,..,N N<=w
            for i in range(0, w+1):
                d2u = img2_descriptor[h,i]
                d = np.linalg.norm(d1u-d2u)
                d_list.append(d)
            disparity[h,w] = np.abs(np.argmin(d_list)-w)
    print("disparity matrix is computed.")
    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int(((-img_width * el[0] - el[2]) / el[1])[0]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('left.bmp', 1)
    img_right = cv2.imread('right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
