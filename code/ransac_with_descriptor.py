#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
#import ipdb

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

def ransac(kp_1, kp_2, ratio_threshold = 0.2, dist_threshold = 2, inlier_tolerance= 1, threshold_inliers=0, max_iter=1000):
    
    homog = 0
    
    #build arrays out of the descriptors
    d_ar_1 = form_descriptor_array(kp_1)
    d_ar_2 = form_descriptor_array(kp_2)

    #match the descriptors; returns index of matching keypoints
    matches_id_1, matches_id_2 = keypoint_matching(d_ar_1, d_ar_2, ratio_threshold, dist_threshold)
#    matches_id_1 = [0,1,6,86]
#    matches_id_2 = [0,1,8,19]

    assert len(matches_id_1) > 4, "Not enough matching keypoints"

    #update the keypoints and descriptor arrays to only have those keypoints
    matching_kps1 = []
    matching_kps2 = []
    cords1 = []
    cords2 = []

    for pt in matches_id_1:
        matching_kps1.append(kp_1[pt])
        cords1.append((kp_1[pt][3], kp_1[pt][2]))
    for pt in matches_id_2:
        
        matching_kps2.append(kp_2[pt])
        cords2.append((kp_2[pt][3], kp_2[pt][2]))

    #form arrays of all the points
    cords1 = np.insert(np.array(cords1).reshape(len(cords1), 2), 2, 1, axis=1).T
    cords2 = np.insert(np.array(cords2).reshape(len(cords2), 2), 2, 1, axis=1).T
    
    #do ransac
    best_error = 1000000
    best_num_inliers = 0
#    import ipdb
    for iteration in range(max_iter):
        
        #randomly sample keypoints
        img_1_pts = random.sample(matches_id_1, 4)

        #get the points that they map to
        selected_cords_1 = []
        selected_cords_2 = []
    
        for xy in img_1_pts:
            selected_cords_1.append((kp_1[xy][3], kp_1[xy][2]))
            selected_cords_2.append((kp_2[matches_id_2[matches_id_1.index(xy)]][3], kp_2[matches_id_2[matches_id_1.index(xy)]][2]))
        
        source_points =np.array(selected_cords_1).reshape(4, 2)
        target_points =np.array(selected_cords_2).reshape(4, 2)
        c1 = np.insert(np.array(selected_cords_1).reshape(4, 2), 2, 1, axis=1).T
        c2 = np.insert(np.array(selected_cords_2).reshape(4, 2), 2, 1, axis=1).T
        #get the homography for the sample points
        homography = get_homography(source_points, target_points)     

        #get the projection of all the points using the homography
        projection = np.dot(homography, cords1)

        if np.count_nonzero(projection[-1] == 0) > 0:
            continue #skip that homography becuase divide by zero

        projection = projection/projection[-1]
#        print()
#        print(cords1)
        
        x = np.dot(homography, c1)
  
        x = np.dot(homography, cords1)
#        print("all target")
#        print(cords2)
#        print("al projection")
#        print(x/x[-1])
    
        #determine the error produced in the projection
        error = np.sqrt(np.sum(np.square(cords2 - projection), axis=0))
#        print("error")
#        print(error)
#        print(error.sum())
#        ipdb.set_trace()
        #test if these four points are better than others
        if np.count_nonzero(error < inlier_tolerance) > threshold_inliers:
            
            
            
            num_inliers = np.count_nonzero(error < inlier_tolerance)
            threshold_inliers = num_inliers
            print("The homography has " + str(int(num_inliers/len(matching_kps1) * 100)) + " percent inliers.")
            #save the homography
            if (error.sum() < best_error) and (best_num_inliers < num_inliers):
                homog = homography
                
                matching_idx = np.where(error < inlier_tolerance)[0]
                

                best_num_inliers = num_inliers
                print("The best homography has " + str(int(num_inliers/len(matching_kps1) * 100)) + " percent inliers.")
    
    if type(homog) == int:
        raise ValueError
        return homography, dict(zip(matches_id_1, matches_id_2))
    else:
        
#        print("The best homography:")
#        x = np.dot(homog, cords1)
#        y = np.around(x/x[-1], 0)
#        print("Target Points")
#        print(cords2)
#        print("Projection using Homography")
#        print(y.astype(int))
#        print("best error:")
#        print(np.abs(x-y).sum().astype(int))
        
        
        return homog, dict(zip(matches_id_1, matches_id_2)), matching_idx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def form_descriptor_array(kps):

    """
    Form the descriptors from all the kps into an array.
    """
    
    counter = 0
    for kp in kps:
        if counter == 0:
            out = np.array(kp[6]).reshape(1, 128)
            counter += 1
        else:
            out = np.vstack((out, np.array(kp[6]).reshape(1, 128)))
    return out

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def keypoint_matching(ds1, ds2, ratio_threshold = 0.2, dist_threshold = 2):

    """
    Using the descriptors, match the keypoints.
    """

    rk1 = []
    rk2 = []
    for descrpt in range(ds1.shape[0]):
        
        d = np.abs(ds1[descrpt,:] - ds2).sum(axis = 1)
        
        #threshold according to lowe's criteria
        if d[np.argsort(d)[1]] == 0:
            denom = 0.1
        else: 
            denom = d[np.argsort(d)[1]]
        
        if (d[np.argsort(d)[0]]/denom < ratio_threshold) or (d[np.argsort(d)[0]] == 0): #test that it is a perfect match or far from second match
            if min(d) < dist_threshold: #only match if distance is sufficiently close
#                print("the dist is" + str(min(d)))
                rk1.append(descrpt)
                rk2.append(np.argsort(d)[0])
#            else:
#                print("dist")
#                print(min(d))
    
    print("there are " + str(len(rk1)) +  " matching keypoints.")
    
    return rk1, rk2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def test_homography(h, src_id, trg_id, kps_1, kps_2):
    
    """
    Test how well a homog is working to map from source to target
    """
    
    src = np.array([kps_1[src_id][2], kps_1[src_id][3] , 1]).reshape(3,1)
    trg = np.array([kps_2[trg_id][2], kps_2[trg_id][3] , 1]).reshape(3,1)
    
    projection = np.dot(h, src)
    print("asource")
    print(src)
    print("proj")
    print(projection/projection[-1])
    print("target")
    print(trg)
    
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

def illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, sample_kp_id):

    def return_sample(desc1,desc2,kps_1,kps_2, sample):
    
        c= np.abs(desc1[sample,:] -desc2).sum(axis = 1)
        
        return kps_1[sample], kps_2[np.argsort(c)[0]], np.argsort(c)[0]

    
    #build descriptor arrays from each sample
    desc1 =form_descriptor_array(kps_1)
    desc2 =form_descriptor_array(kps_2)

    #just take a sample point and retun the corresponding keypoint info from both images


    a, b, c =return_sample(desc1, desc2, kps_1, kps_2, sample_kp_id)
#    print("first kp is at" + str(a[3]) + ", " +str(a[2]))
#    print("second kp is at" + str(b[3]) + ", " +str(b[2]))
#    return c
#    plot them
    fig, (ax1, ax2) = plt.subplots(1, 2)
    circle = plt.Circle((a[3], a[2]), 12, color='G', fill = False)
    ax1.add_artist(circle)
    ax1.set_title('Keypoint' + str(sample_kp_id))
    ax1.imshow(image_1)
    ax1.axis('off')
    circle2 = plt.Circle((b[3], b[2]), 12, color='G', fill = False)
    ax2.add_artist(circle2)
    ax2.imshow(image_2)
    ax2.set_title('Keypoint' + str(c))
    ax2.axis('off')
    plt.gray()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    