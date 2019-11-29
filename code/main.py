#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
from keypoint_detector import keypoint_detector
import utility as ut
import ransac_without_descriptor as rs
import cv2
import ransac_with_descriptor as rsc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Get keypoints for the mountain images
"""

#image_1 = plt.imread(r'../images/mountain_1.png')[:,:,0]
#mt1 = keypoint_detector(image_1, number_of_octaves = 8, number_of_k_per_octave = 12)
#mt1.get_keypoints()
#kps_1 = mt1.keypoints_scaled
##
#image_2 = plt.imread(r'../images/mountain_2.png')[:,:,0]
#mt2 = keypoint_detector(image_2, number_of_octaves = 8, number_of_k_per_octave = 12)
#mt2.get_keypoints()
#kps_2 = mt2.keypoints_scaled
##
#h, matching_keypoints, a = rs.ransac(kps_1, kps_2, dist_threshold=20, ratio_threshold=0.99)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)

#ut.plot_keypoints(mt1.original_image, kps_1, save = False, save_name='mountain_1_kp')
#ut.plot_keypoints(mt2.original_image, kps_2, save = False, save_name='mountain_2_kp')
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, 2)

#for p in matching_keypoints.keys():
#    rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
lena left and right
"""

#image_1 = plt.imread(r'../images/lena_right.tif')#[:,:,0]
#det1 = keypoint_detector(image_1, number_of_octaves = 4, number_of_k_per_octave = 12)
#det1.get_keypoints()
#kps_1 = det1.keypoints_scaled
#
#image_2 = plt.imread(r'../images/lena_left.tif')
#det2 = keypoint_detector(image_2, number_of_octaves = 4, number_of_k_per_octave = 12)
#det2.get_keypoints()
#kps_2 = det2.keypoints_scaled
##
#h, matching_keypoints, idx = rsc.ransac(kps_1, kps_2, ratio_threshold = 2)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)
plt.imshow(image_2)
plt.axis('off')


#point = 8
#rs.test_homography(h, point, matching_keypoints[point], kps_1, kps_2)

#h, matching_keypoints = rsc.ransac(kps_1, kps_2, inlier_threshold = 0.1, max_iter = 10000, inlier_percent= 99)
#result = rsc.warp(h, image_1, image_2)
#rsc.show_result(image_1, image_2, result)
#
#for p in matching_keypoints.keys():
#    if p < 10:
#        rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)

#rsc.test_homography(h, 0, matching_keypoints, kps_1, kps_2)
#ut.plot_keypoints(det1.original_image, kps_1, save = False, save_name='mountain_1_kp')
#ut.plot_keypoints(det2.original_image, kps_2, save = False, save_name='mountain_2_kp')
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,92)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
shape left and right
"""

#image_1 = plt.imread(r'../images/right_shape.png')[:,:,0]
#det1 = keypoint_detector(image_1, number_of_octaves = 4, number_of_k_per_octave = 12)
#det1.run_localisation = False
#det1.get_keypoints()
#kps_1 = det1.keypoints_scaled
##
#image_2 = plt.imread(r'../images/left_shape.png')[:,:,0]
#det2 = keypoint_detector(image_2, number_of_octaves = 4, number_of_k_per_octave = 12)
#det2.run_localisation = False
#det2.get_keypoints()
#kps_2 = det2.keypoints_scaled
#
#h, matching_keypoints = rsc.ransac(kps_1, kps_2, inlier_threshold = 0.5, max_iter = 1000, inlier_percent= 95)
#result = rsc.warp(h, image_1, image_2)
#rsc.show_result(image_1, image_2, result)


#h, matching_keypoints, matching_idx = rs.ransac(kps_1, kps_2, dist_threshold =24, ratio_threshold=1)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)
#
#for p in matching_keypoints.keys():
#    if p < 10:
#    rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)
##
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,89)
#
#point = 278
#rs.test_homography(h, point, matching_keypoints[point], kps_1, kps_2)

##ut.plot_keypoints(det1.original_image, kps_1, save = False, save_name='mountain_1_kp')
#ut.plot_keypoints(det2.original_image, kps_2, save = False, save_name='mountain_2_kp')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
shape left and right rotated
"""

#image_1 = plt.imread(r'../images/box.png')[:,:,0] * 255
#det1 = keypoint_detector(image_1, number_of_octaves = 6, number_of_k_per_octave = 12)
#det1.run_localisation = False
#det1.get_keypoints()
#kps_1 = det1.keypoints_scaled

#image_2 = plt.imread(r'../images/box2.png')[:,:,0] * 255
#det2 = keypoint_detector(image_2, number_of_octaves = 6, number_of_k_per_octave = 12)
#det2.run_localisation = False
#det2.get_keypoints()
#kps_2 = det2.keypoints_scaled

#ut.plot_keypoints(det1.original_image, kps_1, save = False, save_name='mountain_1_kp')
#ut.plot_keypoints(det2.original_image, kps_2, save = False, save_name='mountain_2_kp')
#h, matching_keypoints = rsc.ransac(kps_1, kps_2, inlier_threshold = 0.5, max_iter = 1000, inlier_percent= 90)

#h, matching_keypoints, matching_idx = rs.ransac(kps_1, kps_2, ratio_threshold=2, dist_threshold=17)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)
#
#for p in matching_keypoints.keys():
#    if p < 10:
#        rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, p)
#
#h, matching_keypoints = rsc.ransac(kps_1, kps_2, inlier_threshold = 0.5, max_iter = 1000, inlier_percent= 95)
#result = rsc.warp(h, image_1, image_2)
#rsc.show_result(image_1, image_2, result)


#x1 = kps_1[0][6].reshape(16,8)
#x2 = kps_2[0][6].reshape(16,8)
#np.abs(x1 -x2).sum()
#show the matches on a graph
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, 98)
#47 708
#16 708
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
book
"""

image_1 = plt.imread(r'../images/book1.jpg')
det1 = keypoint_detector(image_1, number_of_octaves = 6, number_of_k_per_octave = 12)
det1.get_keypoints()
kps_1 = det1.keypoints_scaled

image_2 = plt.imread(r'../images/book2.jpg')
det2 = keypoint_detector(image_2, number_of_octaves = 6, number_of_k_per_octave = 12)
det2.get_keypoints()
kps_2 = det2.keypoints_scaled

h1, matching_keypoints1, idx = rsc.ransac(kps_1, kps_2, max_iter=10000,ratio_threshold=5, dist_threshold=8, inlier_tolerance = 1)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)
#
#h, matching_keypoints= rsc.ransac(kps_1, kps_2, max_iter=10000, inlier_threshold = 1)
#
#for p in matching_keypoints.keys():
#    if p<10:
#        rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)
#
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, 50)
#
#ut.plot_keypoints(det1.original_image, kps_1)
#ut.plot_keypoints(det2.original_image, kps_2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Beach
"""

#image_1 = plt.imread(r'../images/carmel-00.png') *255
#det1 = keypoint_detector(image_1, number_of_octaves = 6, number_of_k_per_octave = 12)
#det1.get_keypoints()
#kps_1 = det1.keypoints_scaled
##
#image_2 = plt.imread(r'../images/carmel-01.png') * 255
#det2 = keypoint_detector(image_2, number_of_octaves = 6, number_of_k_per_octave = 12)
#det2.get_keypoints()
#kps_2 = det2.keypoints_scaled
##
#h, matching_keypoints, idx = rsc.ransac(kps_1, kps_2, max_iter=10000,ratio_threshold=5, dist_threshold=8, inlier_tolerance = 1)
#result = rs.warp(h, image_1, image_2)
#rs.show_result(image_1, image_2, result)
#
#h_, matching_keypoints_= rs.ransac(kps_1, kps_2, max_iter=10000, inlier_threshold = 1)
#
#plt.imshow(result)
#plt.axis('off')
#plt.gray()
#
#ut.plot_keypoints(det1.original_image, kps_1)
#ut.plot_keypoints(det2.original_image, kps_2)



#for p in matching_keypoints.keys():
#    if p<100:
#        rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)
#
#rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, 12)
##
#ut.plot_keypoints(det1.original_image, kps_1)
#ut.plot_keypoints(det2.original_image, kps_2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Beach
"""

image_1 = (plt.imread(r'../images/bryce_left_01.png') *255).astype(np.uint8)
det1 = keypoint_detector(image_1, number_of_octaves = 6, number_of_k_per_octave = 12)
det1.get_keypoints()
kps_1 = det1.keypoints_scaled
#
image_2 = (plt.imread(r'../images/bryce_right_01.png') *255).astype(np.uint8)
det2 = keypoint_detector(image_2, number_of_octaves = 6, number_of_k_per_octave = 12)
det2.get_keypoints()
kps_2 = det2.keypoints_scaled
#
h1, matching_keypoints1, idx = rsc.ransac(kps_2, kps_1, max_iter=10000,ratio_threshold=5, dist_threshold=8, inlier_tolerance = 1)
result = rs.warp(h1, image_2, image_1)
rs.show_result(image_1, image_2, result)



h_, matching_keypoints_= rs.ransac(kps_1, kps_2, max_iter=10000, inlier_threshold = 1)

plt.imshow(result)
plt.axis('off')
plt.gray()

ut.plot_keypoints(det1.original_image, kps_1)
ut.plot_keypoints(det2.original_image, kps_2)



for p in matching_keypoints.keys():
    if p<100:
        rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2,p)

rs.illustrate_keypoint_matching(image_1, image_2, kps_1, kps_2, 12)
#
ut.plot_keypoints(det1.original_image, kps_1)
ut.plot_keypoints(det2.original_image, kps_2)

