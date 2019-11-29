import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_homography(points1, points2):

    """
	Determine the homography between two sets of points
    Returns 3x3 trasnformation matrix
    Input must be 2d array, each row is a set of points (x, y), of shape (4, 2)
    where we are testing 4 points with two coords each
    """

    num_points = 4
    trans = np.zeros((2*num_points, 9))
    for p in range(num_points):
        pp = np.array([points1[p][0], points1[p][1], 1])
        trans[2*p] = np.concatenate(([0, 0, 0], -pp, points2[p][1]*pp))
        trans[2*p + 1] = np.concatenate((pp, [0, 0, 0], -points2[p][0]*pp))
    U, D, V = np.linalg.svd(trans)
    homog = V[8].reshape(3, 3)

    return homog/homog[-1,-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ransac(kps_1, kps_2, inlier_threshold=0.5, max_iter=1000, inlier_percent = 80):
    
    best_num_inliers = 0

    #x, y for all kps
    kp1 = np.ones((3, len(kps_1)))
    kp2 = np.ones((3, len(kps_2)))
    for kp in range(len(kps_1)):
        kp1[0, kp] = kps_1[kp][3]
        kp1[1, kp] = kps_1[kp][2]
        
    for kp in range(len(kps_2)):
        kp2[0, kp] = kps_2[kp][3]
        kp2[1, kp] = kps_2[kp][2]    
    
    best_num_inliers = 0
    iteration = 0
    converged = False
    percent_inliers = 0
    matching_keypoints = {}
    while not converged:
        
        if iteration%100 == 0:
            print(str((int((iteration+1)/max_iter * 100))) + "% complete")
        
        source_points = np.ones((4, 2)) #to create homography
        counter = 0 
        ar1 = np.ones((3, 4)) #to test homography
        p1 = []
        while len(p1) != 4:
            n = random.randint(0, len(kps_1) - 1)
            if n not in p1:
                p1.append(n)
                ar1[0, counter] = kp1[0,n].copy()
                ar1[1, counter] = kp1[1,n].copy()
                source_points[counter,0] = kp1[0,n].copy()
                source_points[counter,1] = kp1[1,n].copy()
                counter += 1
                
        target_points = np.ones((4, 2))
        counter = 0
        ar2 = np.ones((3, 4)) #to test homography
        p2 = []
        while len(p2) != 4:
            n = random.randint(0, len(kps_2)  - 1)
            if n not in p2:
                p2.append(n)
                ar2[0, counter] = kp2[0,n].copy()
                ar2[1, counter] = kp2[1,n].copy()
                target_points[counter,0] = kp2[0,n].copy()
                target_points[counter,1] = kp2[1,n].copy()
                counter += 1

        #get the homography for the sample points
        homography = get_homography(source_points, target_points)     
        
        #test the homography
        targt = np.dot(homography, ar1)
        targt = targt/targt[-1]

        if (targt - ar2).sum() > 2:
            continue #in some cases the homog got two of the same points. just skip those
#        print("projected target")
#        print(np.around(targt, 0))
#        print("actual target")
#        print(ar2)

        #get the projection of all the points using the homography
        projection = np.dot(homography, kp1)
        projection = projection/projection[-1]
        
        if np.count_nonzero(projection[-1] == 0) > 0:
            continue #skip that homography becuase divide by zero
                
        #count the inliers
        num_inliers = 0
        inliers1 = []
        inliers2 = []
        for pt in range(projection.shape[1]):
            diffs = np.abs(projection[:,pt][None].T - kp2).sum(axis = 0)
            closest_dist = diffs.min()
            
            if closest_dist < inlier_threshold:
                num_inliers += 1
                inliers1.append(pt)
                inliers2.append(np.argmin(diffs))
        
        #decide if its the best homog
        if num_inliers > best_num_inliers:
            matching_keypoints = dict(zip(inliers1, inliers2))
            best_homography = homography.copy()
            best_num_inliers = num_inliers
            percent_inliers = int(num_inliers/len(kps_1) * 100)
            print("Best homography has " + str(num_inliers) + " inliers which is " + str(percent_inliers) + "%.")
        
        iteration += 1
        if percent_inliers >= inlier_percent or iteration >= max_iter:
            converged = True
        
    return best_homography, matching_keypoints

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def warp(h, image_1, image_2, fix_background =True):
    
    """
    Warp an image and make pretty
    """

    dst = cv2.warpPerspective(image_1,h,(image_2.shape[1] + image_1.shape[1], image_2.shape[0]))
    dst[0:image_2.shape[0],0:image_2.shape[1]] = image_2
        
    def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop top
        if not np.sum(frame[:,0]):
            return trim(frame[:,1:])
        #crop top
        if not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])
        return frame
    if fix_background:
        return trim(dst)
    else:
        return dst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def show_result(image_1, image_2, result):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    
    ax1.imshow(image_1) 
    ax1.set_title('Image 1')
    ax1.axis('off')
    ax2.set_title('Image 2')
    ax2.imshow(image_2)
    ax2.axis('off')
    ax3.set_title('Warped')
    ax3.imshow((result))
    ax3.axis('off')
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def test_homography(h, src_id, matching_keypoints, kps_1, kps_2):
    
    """
    Test how well a homog is working to map from source to target
    """
    
    src = np.array([kps_1[src_id][3], kps_1[src_id][2] , 1]).reshape(3,1)
    trg = np.array([kps_2[matching_keypoints[src_id]][3], kps_2[matching_keypoints[src_id]][2] , 1]).reshape(3,1)
    
    projection = np.dot(h, src)
    print("asource")
    print(np.around(src, 1))
    print("proj")
    print(np.around(projection/projection[-1], 1))
    print("target")
    print(np.around(trg, 1))
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       
#error = np.sqrt(np.sum(np.square(kp2 - projection), axis=0))

        
        
        
        
        