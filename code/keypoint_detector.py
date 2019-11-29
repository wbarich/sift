#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import scipy.ndimage as nd
import scipy.ndimage.filters as filters
import scipy.stats as st
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class keypoint_detector():

    """
    This class takes an input image and locates the scale invariant keypoints
    on it.
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, image, number_of_octaves = 1, number_of_k_per_octave = 4, extrema_keep_ties = False, extrema_tie_sensitivity = 1):

        self.original_image = image.copy() #copy the orignal that we will not alter
        self.image = image #the original image that will be transformed through up and down scaling
        self.get_num_channels() #determine how many channesls there are in the image

        self.sigma = 1
        self.k = 2**0.5
        self.gaussian_kernel_size = 3 #size of the kernel with which we can build the dogs
        self.number_of_octaves = number_of_octaves #number of octaves
        self.number_of_k_per_octave = number_of_k_per_octave #number of k levels in an octave

        self.initial_blur = True #whether or not to blur the image at the start
        self.run_localisation = True
        
        self.extrema_search_radius = 3 #radius of the area that is searched to identify extrema
        self.extrema_keep_ties = extrema_keep_ties #whether or not to keep extrema that are tied with others in the search radius
        self.extrema_tie_sensitivity = extrema_tie_sensitivity #how sensitive we are about keeping extrema ties #1 == Safe; 4 == Medium; #8 == Does Nothing

        self.blurred_images = {} #contains blurred images stored by octave and k pair
        self.d = {} #dictionary of stacked dogs, key is octave number
        self.keypoints = {} #indicies of the keypoints (octave, level, x, y)

        self.contrast_threshold = 0.03 #contrast acceptance criteria
        self.curvature_threshold = 10 #curvature acceptance criteria

        self.contrast_rejected_keypoints = {} #keypoints that are rejected becasuse of contrast
        self.edge_rejected_keypoints = {} #keypoints that are rejected because they are on an edge

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_num_channels(self):

        """
        Determine the number of channels in the image
        Assume that the r,g,b bands are in the last dimension
        If the image only has 1 channel, reshape the image with an additional channel over which we can iterate.
        """

        if self.original_image.ndim == 2:
            self.num_channels = 1
            self.image = self.image.reshape(self.image.shape[0], self.image.shape[1], 1)
        else:
            self.num_channels = self.original_image.shape[2]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def resize_image(self, image, factor):

        """
        Resize an image to build the image pyramid.
        """

        resized = resize(image, (int(image.shape[0] * factor), int(image.shape[1] * factor)), anti_aliasing = True)

        if resized.max() < 1:
            return (resized * 255).astype(int)
        else:
            return resized

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def apply_initial_blur(self, image):

        """
        Blur the original image to reduce the noise.
        """

        if self.initial_blur:

            print("Applying initial blur...")

            kernel = self.build_gaussian_filter(3, 1.6)
            return nd.convolve(image, kernel)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_image_pyramid(self, image):

        """
        Build an image pyramid.
        The number of images in the pyramid are equal to the number of octaves.
        """

        pyramid = {}
        for octave in range(self.number_of_octaves):
            pyramid[octave] = self.resize_image(image, 1/(2**(octave - 1)))

        return pyramid

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_gaussian_filter(self, size, sigma):

        """
        Build a gaussian filter according to a specified sigma.
        """

        if type(size) == tuple:
            linear_range_1 = np.linspace(-sigma, sigma, size[0]+1)
            kernel_1d = np.diff(st.norm.cdf(linear_range_1))
            linear_range_2 = np.linspace(-sigma, sigma, size[1]+1)
            kernel_1d_2 = np.diff(st.norm.cdf(linear_range_2))
            kernel_2d = np.outer(kernel_1d, kernel_1d_2)
            return kernel_2d/kernel_2d.sum()
        else:
            linear_range = np.linspace(-sigma, sigma, size+1)
            kernel_1d = np.diff(st.norm.cdf(linear_range))
            kernel_2d = np.outer(kernel_1d, kernel_1d)
            return kernel_2d/kernel_2d.sum()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def apply_gaussians(self, pyramid):

        """
        For a given octave, loop through a set of gaussian kenrels and save the resulting images.
        """

        blurred_images = {}

        #iterate over the octaves
        for octave in range(self.number_of_octaves):

            #iterate over the sigmas
            for k in range(self.number_of_k_per_octave):
                kernel = self.build_gaussian_filter(self.gaussian_kernel_size, self.sigma * ((self.k + 1)**(k)))

                assert kernel.ndim == pyramid[octave].ndim, "Kernel and image have different dimensions" + str(kernel.ndim) + " " + str(pyramid[octave].ndim)
                blurred_images[octave, k] = nd.correlate(pyramid[octave], kernel)

        return blurred_images

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_difference_of_gaussians(self, blurred_images):

        """
        Using the blurred images calculate the difference of gaussians (dog).
        """

        dogs = {}

        #iterate over the blurred images
        for octave in range(self.number_of_octaves):
            for k in range(self.number_of_k_per_octave - 1): #- 1 since the last image has nothing to compare against
                dogs[octave, k] = blurred_images[octave, k] - blurred_images[octave, k + 1]

        return dogs

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def find_extrema(self, image):

        """
        By searching a square neighbourhood around each point of size (search_radius, search_radius, search_radius),
        identify local minima.

        Assume that the images are stacked on the 3rd dimension (1, 2, 3).
        """

        #because we only keep the minima in th emiddle slice
        assert self.extrema_search_radius % 2 == 1, 'The search radius must be an odd number'

        #set up the search neighbourhood
        neighborhood = np.ones((self.extrema_search_radius, self.extrema_search_radius, self.extrema_search_radius), dtype = bool)

        #finding the local minima in the area
        local_minima = (filters.minimum_filter(image, footprint=neighborhood) == image)* 1
        local_maxima = (filters.maximum_filter(image, footprint=neighborhood) == image)* 1

        #now since the filter identified a lot of points that are actually not maxima/minima
        #for example in the case where all the pixels in the neighbourhood are the same all of them are
        #flagged as maxima/minima. So now we eliminate keypoints where there is more than one minima or
        #maxima in a local area

        if not self.extrema_keep_ties:
            original_minima = local_minima.copy()
            original_maxima = local_maxima.copy()
            #print("Before tweaking there were " + str(local_maxima.sum()) + " maxima and " + str(local_minima.sum()) + " local minima.")
            kernel = np.ones((self.extrema_search_radius, self.extrema_search_radius, self.extrema_search_radius))
            local_maxima = nd.convolve(local_maxima, kernel)
            local_minima = nd.convolve(local_minima, kernel)
            local_maxima = np.where(local_maxima > self.extrema_tie_sensitivity, 0, local_maxima)
            local_minima = np.where(local_minima > self.extrema_tie_sensitivity, 0, local_minima)
            local_minima = local_minima * original_minima
            local_maxima = local_maxima * original_maxima
            #print("After tweaking there were " + str(local_maxima.sum()) + " maxima and " + str(local_minima.sum()) + " local minima.")

        #remove minima on the outside layers of the array
        local_minima[:,:,0] = 0
        local_minima[:,:,-1] = 0
        local_minima[0,:,:] = 0
        local_minima[-1,:,:] = 0
        local_minima[:,0,:] = 0
        local_minima[:,-1,:] = 0
        local_maxima[:,:,0] = 0
        local_maxima[:,:,-1] = 0
        local_maxima[0,:,:] = 0
        local_maxima[-1,:,:] = 0
        local_maxima[:,0,:] = 0
        local_maxima[:,-1,:] = 0

        return local_minima, local_maxima

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def identify_initial_keypoints(self, dogs, keypoints):

        """
        Using the local minimum criteria, find the local minima of d.
        D is defined in the same manner as in the paper.
        Structure of each keypoint is [octave, level, x, y]
        """

        minima = {} #each key has the extreme values of an octave
        maxima = {} #each key has the extreme values of an octave
        d = {}

        #for each octave find the keypoints
        for octave in range(self.number_of_octaves):

            #create d, a function that is all the dogs of an octave stacked together
            counter = 0
            for dog in range(self.number_of_k_per_octave - 1):
                if counter == 0:
                    d[octave] = dogs[octave, dog].astype(np.float64)
                else:
                    d[octave] = np.dstack((d[octave], dogs[octave, dog].astype(np.float64)))
                counter+=1

            #for each octave identify the local minima
            minima[octave], maxima[octave] = self.find_extrema(d[octave])

            #create dictionary with the indicies of the keypoints
            keypoints.extend(list(zip([octave] * len(np.nonzero(minima[octave])[2]), np.nonzero(minima[octave])[2], np.nonzero(minima[octave])[0], np.nonzero(minima[octave])[1])))
            keypoints.extend(list(zip([octave] * len(np.nonzero(maxima[octave])[2]), np.nonzero(maxima[octave])[2], np.nonzero(maxima[octave])[0], np.nonzero(maxima[octave])[1])))

        return d, keypoints

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def determine_keypoint_stability(self, d, octave, level, x, y, keypoints, contrast_rejected_keypoints, edge_rejected_keypoints):

        """
        At a particular keypoint, compute the Hessian and Jacobian. Use them to decide whether we should keep a keypoint or not.
        """

        #calculate the elements of the hessian and jacobian
        dx = (d[octave][x + 1, y, level] - d[octave][x - 1, y, level])/2
        dy = (d[octave][x, y + 1, level] - d[octave][x, y - 1, level])/2
        ds = (d[octave][x, y, level + 1] - d[octave][x, y, level - 1])/2.
        dxx = d[octave][x + 1, y, level] - 2 * d[octave][x, y, level] + d[octave][x - 1, y, level]
        dxy = ((d[octave][x + 1, y + 1, level] - d[octave][x - 1, y + 1, level]) - (d[octave][x + 1, y - 1, level] - d[octave][x - 1, y - 1, level]))/4
        dxs = ((d[octave][x + 1, y, level + 1]- d[octave][x - 1, y, level + 1]) - (d[octave][x + 1, y, level - 1] - d[octave][x - 1, y, level - 1]))/4
        dyy = d[octave][x, y + 1, level] - 2 * d[octave][x, y, level] + d[octave][x, y - 1, level]
        dys = ((d[octave][x, y + 1, level + 1] - d[octave][x, y - 1, level + 1]) - (d[octave][x, y + 1, level - 1] - d[octave][x, y - 1, level - 1]))/4
        dss = d[octave][x, y, level + 1] - 2 * d[octave][x, y, level] + d[octave][x, y, level]

        #calculate the jacobian and hessian
        jacobian = np.array([dx, dy, ds])
        hessian = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

        #calculate the offset
        try:
            offset = -np.linalg.inv(hessian).dot(jacobian)
            contrast = d[octave][x, y, level] + 0.5 * jacobian.dot(offset)
        except:
            offset = 0
            contrast = self.contrast_threshold + 1

        #determine if the keypoint meets the contrast threshold
        if abs(contrast) < self.contrast_threshold:
            contrast_rejected_keypoints.append([octave, level, x, y])
            keypoints.remove(tuple((octave, level, x, y)))

        #determine if the keypoint is on an edge
        else:
            values, vectors = np.linalg.eig(hessian)
            if values[0] == 0:
                edge_rejected_keypoints.append([octave, level, x, y])
                keypoints.remove(tuple((octave, level, x, y)))
            else:    
                eigen_ratio = values[1]/values[0]
                curvature_ratio = (eigen_ratio + 1)**2 / eigen_ratio
    
                #determine if the point meets the acceptance criteria
                if curvature_ratio < self.curvature_threshold:
                    edge_rejected_keypoints.append([octave, level, x, y])
                    keypoints.remove(tuple((octave, level, x, y)))

        return keypoints, contrast_rejected_keypoints, edge_rejected_keypoints

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def evaluate_all_keypoints(self, d, keypoints):

        """
        Test whether the keypoints are stable or not using contrast and edge criteria.
        """

        #iterate over each keypoint
        all_keypoints = keypoints.copy()
        contrast_rejected_keypoints = []
        edge_rejected_keypoints = []

        for index, keypoint in enumerate(all_keypoints):

            if index%5000 == 0:
                print("Keypoint evaluation " + str(int(100 * index/len(all_keypoints)) + 1) + "% complete. ")

            keypoints, contrast_rejected_keypoints, edge_rejected_keypoints = self.determine_keypoint_stability(d, keypoint[0], keypoint[1], keypoint[2], keypoint[3], keypoints, contrast_rejected_keypoints, edge_rejected_keypoints)

        print("\n" + "There were " + str(len(contrast_rejected_keypoints)) + " contrast rejected keypoints.")
        print("There were " +  str(len(edge_rejected_keypoints)) + " edge rejected keypoints.")
        print("After localisation there were " + str(len(keypoints)) + " keypoints. \n")

        return keypoints, contrast_rejected_keypoints, edge_rejected_keypoints

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def scale_keypoints_to_original_scale(self):

        """
        Since the kp are all in different scale spaces, we want to scale them
        back to the original scale for viz purposes.
        """

        print("Scaling keypoints to original scale...")

        #scale the keypoints
        keypoints_scaled = []
        for kp in self.keypoints:
            if (int(kp[2] * (2**(kp[0]-1))), int(kp[3] * (2**(kp[0]-1)))) not in keypoints_scaled:
                keypoints_scaled.append((kp[0], kp[1], int(kp[2] * (2**(kp[0]-1))), int(kp[3] * (2**(kp[0]-1))), kp[4], kp[5], kp[6]))

        return keypoints_scaled

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def make_keypoints_all_1_channel(self, keypoints):

        #remove the keypoints that are duplicates (may be duplicate accross scale spaces)
        unique_kps = []
        for kp in keypoints:
            if (kp[2], kp[3]) not in unique_kps:
                unique_kps.append(kp)
        return unique_kps

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_keypoints(self):

        """
        Run all the methods in the class and return the keypoints
        for each channel.
        """

        #identify the keypoints
        for channel in range(self.num_channels):

            print("\n" + "Finding keypoints for channel: " +str(channel) + str("..."))

            keypoints = []
            image = self.image[:,:,channel]
            image = self.apply_initial_blur(image) #applies a gaussian blur to the image if initial_blur is true
            pyramid = self.build_image_pyramid(image) #builds image pyramid
            self.blurred_images[channel] = self.apply_gaussians(pyramid) #blurs each image in pyramid
            dogs = self.build_difference_of_gaussians(self.blurred_images[channel]) #calculates dogs
            d, keypoints = self.identify_initial_keypoints(dogs, keypoints) #finds extrema in dogs
            
            if self.run_localisation:
                self.keypoints[channel], self.contrast_rejected_keypoints[channel], self.edge_rejected_keypoints[channel] = self.evaluate_all_keypoints(d, keypoints) #keypoint localisation
            else:
                self.keypoints[channel] = keypoints.copy()

        #add the colour channel to the keypoint info
        self.keypoints = self.add_channel_info()

        #get the keypoint orientation
        self.orientate_the_keypoints()

        #build the keypoint descriptors
        self.build_descriptors()

        #create nice structure to keep the keypoints
        self.keypoints_scaled = self.scale_keypoints_to_original_scale() #scale keypoints to original scale
        #self.keypoints = self.make_keypoints_all_1_channel(self.keypoints)# not working as not on same scale
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_channel_info(self):

        """
        Add the chanel info to the keypoints
        """
        new_kp = []
        for channel in self.keypoints.keys():
            for kp in self.keypoints[channel]:
                new_kp.append((kp[0], kp[1], kp[2], kp[3], channel))

        return new_kp

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_image_sample(self, img, point, search_radius = 8):

        """
        For a given image return a sample of that image based on a search radius
        """

        #set the bounds of the search area for computing the histogram
        xlim_min = point[0] - int(search_radius/2)
        xlim_max = point[0] + int(search_radius/2)
        ylim_min = point[1] - int(search_radius/2)
        ylim_max = point[1] + int(search_radius/2)

        if xlim_min < 0:
            xlim_min = 0
        if ylim_min < 0:
            ylim_min = 0
        if xlim_max > img.shape[0]:
            xlim_max = img.shape[0]
        if ylim_max > img.shape[1]:
            ylim_max = img.shape[1]
        
        if ylim_max < ylim_min or xlim_max < xlim_min:
            raise ValueError
        
        #get the sample area
        return img[xlim_min: xlim_max, ylim_min: ylim_max]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_sample_orientation(self, sample, number_of_bins, normalize, kp_orientation = 0):

        """
        For a specific point determine the oreintation of a keypoint.
        Weight the neighbour hood of the image based on the sigma used to blur
        the image initially.
        """

        #loop through each point in the area
        theta = []
        bins_orientation = np.arange(-180, 190, 10)

        for x in range(1, sample.shape[0]-1): #dont loop over the boundry points
            for y in range(1, sample.shape[1]-1):

                #calculate the gradient magnitude and direction
                magnitude = np.sqrt(np.square(sample[x + 1, y] - sample[x- 1, y]) + np.square(sample[x, y + 1] - sample[x, y - 1]))
                angle = [np.arctan2((sample[x, y + 1] - sample[x, y - 1]),(sample[x + 1, y] - sample[x- 1, y])) * number_of_bins/2*np.pi]
                magnitude = np.ceil(np.array(magnitude))
                if magnitude == 0: #for the case that all the pixels are zero the magnitude is zero. so force it to be one
                    magnitude = 1
                angle = angle *  int(magnitude)#weight by magnitude
                theta.extend(angle)

        if normalize: #in the case that we are returning the descriptor

            #rotate the features according to the orientation of the kp
            theta = np.array(theta) + bins_orientation[kp_orientation]
            if theta[0] < -180:
                theta = theta + 360
            elif theta[0] > 180:
                theta = theta - 360

            #bins for histogram
            bins = np.arange(-180, 200, 45)
            hist = np.histogram(theta, bins)[0]
            return hist/hist.sum()

        else: #in the case we are returning the orientation of keypoint
            bins = np.arange(-180, 190, 10)
            hist = np.histogram(theta, bins)[0]
            return np.argmax(hist)

    def smpl_ort(self, sample, number_of_bins, normalize, kp_orientation = 0):
        
        theta = []
        
        for x in range(1, sample.shape[0]-1): #dont loop over the boundry points
            for y in range(1, sample.shape[1]-1):
                
                #calculate the gradient magnitude and direction
                magnitude = np.sqrt(np.square(sample[x - 1, y] - sample[x + 1, y]) + np.square(sample[x, y + 1] - sample[x, y - 1]))
                
                angle_in_degrees = np.arctan2(sample[x - 1, y] - sample[x + 1, y],sample[x, y + 1] - sample[x, y - 1])/np.pi*180
                #correct negative angles
                if angle_in_degrees < 0:
                    angle_in_degrees += 360
                
          
                
                
                bin_ = np.around(np.ceil(angle_in_degrees/360 * number_of_bins), 0)
                if bin_ == 0:
                    bin_ = number_of_bins
                bin_you_are_in = [bin_]
                
#                print("sample")
#                print(np.around(sample[x - 1: x + 2, y- 1:y+2], 1))
#                print("vert cahgne")
#                print(sample[x - 1, y] - sample[x+ 1, y])
#                print("horz change")
#                print(sample[x, y + 1] - sample[x, y - 1])
#                print("angle")
#                print(angle_in_degrees)
#                print("bin of descriptor")
#                print(bin_)
#                print("bin of kp")
#                print(kp_orientation)
#                print("adjusted kp orientation")
#                print(np.around(kp_orientation/36*8,0))
                
                
#                if normalize:
#                    import ipdb
#                    ipdb.set_trace()
                    
                magnitude = np.ceil(np.array(magnitude))
                if magnitude == 0: #for the case that all the pixels are zero the magnitude is zero. so force it to be one
                    magnitude = 1
                bin_you_are_in = bin_you_are_in *  int(magnitude)#weight by magnitude
                
                if normalize: #in the case that we are returning the descriptor
                    #rotate the features according to the orientation of the kp
                    bin_you_are_in = np.array(bin_you_are_in) - np.array(np.around(kp_orientation/36*8,0))
                    
#                    import ipdb
#                    ipdb.set_trace()
                    if bin_you_are_in[0] < 0:
                        bin_you_are_in = bin_you_are_in + 8
                        bin_you_are_in = bin_you_are_in.tolist()
                    if bin_you_are_in[0] == 0:
                        bin_you_are_in = [8] * int(magnitude)
#                    print("final bin")
#                    print(bin_you_are_in)
                    theta.extend(bin_you_are_in)
                else:
                    theta.extend(bin_you_are_in)
                       
        if normalize:         
            
            bins = np.arange(1,10, 1)
            hist = np.histogram(theta, bins)[0]
#            print("here is the hist" + str(hist))
#            import ipdb
#            ipdb.set_trace()
            if hist.sum() == 0:
                raise ValueError
#                import ipdb
#                ipdb.set_trace()
            
            return hist/hist.sum()
                    
                    
        else: #in the case we are returning the orientation of keypoint
#            ipdb.set_trace()
            bins = np.arange(1, 38, 1)
            hist = np.histogram(theta, bins)[0]
#            ipdb.set_trace()
#            import ipdb
#            ipdb.set_trace()
            
#            print("here is the hist" + str(hist))
            return np.argmax(hist) + 1
        
        
        
        
        
        
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def orientate_the_keypoints(self):

        """
        For each keypoint, determine the orientation of the keypoint.
        Modify the keypoints so that they now also have the orientation.
        kp structure is [octave, level, x, y, channel, orientation]
        """

        new_kps = []
        for kp in self.keypoints:
            sigma_used = self.sigma * kp[1]**(kp[1] - 1) #according to the blur images method
            local_area_around_kp = self.get_image_sample(self.blurred_images[kp[4]][kp[0], kp[1]], (kp[2], kp[3]))
            
            if local_area_around_kp.shape == (8,8): #only keep the kp if it has 8 x 8 area around it
            
                kernel = self.build_gaussian_filter((local_area_around_kp.shape[0], local_area_around_kp.shape[1]), sigma_used * 1.5)
                kernel += 1
                local_area_around_kp = local_area_around_kp * kernel #weight the image sample with a gaussian
                orientation = self.get_sample_orientation(local_area_around_kp, number_of_bins = 36, normalize = False)
                orientation = self.smpl_ort(local_area_around_kp, number_of_bins = 36, normalize = False)
#                print("the orientation is " + str(orientation))
                new_kps.append((kp[0], kp[1], kp[2], kp[3], kp[4], orientation))
                
        self.keypoints = new_kps

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_descriptors(self):

        """
        Build the keypoint descriptors.
        """

        new_kps = []
        for kp in self.keypoints:
            
            sigma_used = self.sigma * kp[1]**(kp[1] - 1) #according to the blur images method
            local_area_around_kp = self.get_image_sample(self.blurred_images[kp[4]][kp[0], kp[1]], (kp[2], kp[3]), search_radius = 16) #get 16 x 16 neighbourhood
#            kernel = self.build_gaussian_filter((local_area_around_kp.shape[0], local_area_around_kp.shape[1]), sigma_used * 1.5)
#            kernel += 1
#            local_area_around_kp = local_area_around_kp * kernel #weight the image sample with a gaussian
            
            #if the kp is a border point we dont keep it
            if local_area_around_kp.shape == (16,16):
                
                #make sub regions
                descriptor = []
                for x in range(4):
                    for y in range(4):
                        data_to_analyze = local_area_around_kp[(4*x):(4*x)+4,(4*y):(y*4)+4].copy()
                        #in some cases we cannot get a 4x4 area because there is not enough data to sample from the picture (eg corners)
                        if (data_to_analyze.shape[0] == 0) or (data_to_analyze.shape[1] == 0) or (data_to_analyze.shape[0] < 3) or (data_to_analyze.shape[1] < 3):
                            descriptor.append(np.ones(8))
                        elif np.unique(data_to_analyze).shape == (1,):
                            descriptor.append((np.ones(8)))#if all the values are the same there is no gradient so make all zeros                
                        else:
                            #descriptor.append(self.get_sample_orientation(data_to_analyze, number_of_bins = 8, normalize = True, kp_orientation = kp[5]))
                            descriptor.append(self.smpl_ort(data_to_analyze, number_of_bins = 8, normalize = True, kp_orientation = kp[5]))
    
                new_kps.append((kp[0], kp[1], kp[2], kp[3], kp[4], kp[5], np.concatenate(descriptor, axis = 0)))

        self.keypoints = new_kps

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
