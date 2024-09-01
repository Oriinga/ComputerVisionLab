from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

vertical_prewitt = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])
horizontal_prewitt = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

laplacian = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])


class Stat_Classifier:

    def __init__(self,image) -> None:
        self.image = image
        pass
    def classify(self,validation_features,fg_features,bg_features,mask,image):
        """
        Multiplies two numbers and returns the result.

        Args:
            image_features : The features for whole image.

        Returns:
            vector (2d): the predictions per pixel.
        """
        
        fg_feature_matrix = np.stack(fg_features, axis=-1)
        fg_mean_vector = np.mean(fg_feature_matrix, axis=0)
        fg_cov_matrix = np.cov(fg_feature_matrix, rowvar=False)
        print("fg mean shape:",fg_mean_vector.shape)
        print("fg cov shape:", fg_cov_matrix.shape)
        # print(np.sum(fg_mean_vector))

        #make each feature a row in the matrix
        # print(fg_features.shape)
        bg_feature_matrix = np.stack(bg_features, axis=-1)
        bg_mean_vector = np.mean(bg_feature_matrix, axis=0)
        bg_cov_matrix = np.cov(bg_feature_matrix, rowvar=False)
        print("bg mean shape:",bg_mean_vector.shape)
        print("bg cov shape",bg_cov_matrix.shape)
        
    
        reshaped_features = validation_features.T
        
        
        ####### vector of predictions #######
        probabilities = self.foreground_given_pixel(reshaped_features, fg_mean_vector, fg_cov_matrix, bg_mean_vector, bg_cov_matrix,mask,image)
        
        height, width = self.image.shape[0], self.image.shape[1]
        
        predictions_reshaped = probabilities.reshape(height, width)
    
        return predictions_reshaped
    
    def foreground_given_pixel(self,x,fg_mean, fg_cov, bg_mean, bg_cov,mask,image):
        """
        Args:
            mask (2d array): Remember to binarize it.
            image (type):the original image.

        Returns:
            type: probability.
        """
        N = image.shape[0]*image.shape[1]
        N_fg = np.sum(mask)
        N_bg = N - N_fg
        
        numerator = multivariate_normal.pdf( x, mean = fg_mean, cov= fg_cov, allow_singular=True) * (N_fg)
        denominator = multivariate_normal.pdf(x, mean=fg_mean, cov=fg_cov, allow_singular=True)*N_fg \
                    + multivariate_normal.pdf( x, mean= bg_mean, cov= bg_cov, allow_singular=True) * (N_bg)
        small_value = 1e-10  # You can adjust the small value if needed
        denominator = np.where(denominator == 0, small_value, denominator)
        probability = numerator/denominator
        return probability
    
    def getFeatures(self,training_img, mask, show_plot=False):
        """
        Parameters:
            training_img (2d array): training image.
            mask (type): binarized image.

        Returns:
            type: Flattened features.
        """
        if(type(mask[0][0]) != np.bool_):
            binary_mask = mask >128

        vertical_prewitt = np.array([
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
        ])
        horizontal_prewitt = np.array([
            [1,0,-1],
            [1,0,-1],
            [1,0,-1]
        ])

        laplacian = np.array([
            [0,-1,0],
            [-1,4,-1],
            [0,-1,0]
        ])

        gauss = GaussFilter(49, 10**0.5)
        lgauss =  LoG(49, 10**0.5)
        dgauss = DoG(49,5,10)

        binary_mask = mask>128
        #plt.imshow(binary_mask)
        
        #add dimensions
        # print(binary_mask.shape)
        hsv_training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
        v,s,h = cv2.split(hsv_training_img)
        h, s,v = h*binary_mask, s*binary_mask, v*binary_mask
        # print(h.shape)
        b,g,r = cv2.split(training_img)
        r,g,b = r*binary_mask, g*binary_mask, b*binary_mask


        # get vertical prewitt for separated channels

        vert_prewitt_r = cv2.filter2D(src=r, ddepth=-1, kernel=vertical_prewitt)
        vert_prewitt_g = cv2.filter2D(src=g, ddepth=-1, kernel=vertical_prewitt)
        vert_prewitt_b = cv2.filter2D(src=b, ddepth=-1, kernel=vertical_prewitt)
        # get horizontal prewitt for separated channels

        hori_prewitt_r = cv2.filter2D(src=r, ddepth=-1, kernel=horizontal_prewitt)
        hori_prewitt_g = cv2.filter2D(src=g, ddepth=-1, kernel=horizontal_prewitt)
        hori_prewitt_b = cv2.filter2D(src=b, ddepth=-1, kernel=horizontal_prewitt)
        # get Laplacian for separated channels

        laplace_r = cv2.filter2D(src=r, ddepth=-1, kernel=laplacian)
        laplace_g = cv2.filter2D(src=g, ddepth=-1, kernel=laplacian)
        laplace_b = cv2.filter2D(src=b, ddepth=-1, kernel=laplacian)

        # get gaussian for seperate channels
        gauss_r = cv2.filter2D(src=r, ddepth=-1, kernel = gauss)
        gauss_g = cv2.filter2D(src=g, ddepth=-1, kernel = gauss)
        gauss_b = cv2.filter2D(src=b, ddepth=-1, kernel = gauss)

        # get log of gaussian for seperate channels
        l_gauss_r = cv2.filter2D(src=r, ddepth=-1, kernel = lgauss)
        l_gauss_g = cv2.filter2D(src=g, ddepth=-1, kernel = lgauss)
        l_gauss_b = cv2.filter2D(src=b, ddepth=-1, kernel = lgauss)

        # get log of gaussian for seperate channels
        d_gauss_r = cv2.filter2D(src=r, ddepth=-1, kernel = dgauss)
        d_gauss_g = cv2.filter2D(src=g, ddepth=-1, kernel = dgauss)
        d_gauss_b = cv2.filter2D(src=b, ddepth=-1, kernel = dgauss)

        # get LBPs for seperate channels
        lbp_r = getLBPs(r)
        lbp_g = getLBPs(g)
        lbp_b = getLBPs(b)

        # get Harr for seperate channels and sizes
        integral_images = [cv2.integral(training_img[:,:,i]) for i in range(3)]
        haar4 = apply_haar_filter(integral_images,4)
        haar4_r = haar4[0]
        haar4_g = haar4[1]
        haar4_b = haar4[2]
        #haar4_r, haar4_g, haar4_b = haar4[0],haar4[1],haar4[2]
        haar8 = apply_haar_filter(integral_images,8)
        haar8_r, haar8_g, haar8_b = haar8[0],haar8[1],haar8[2]
        haar16 = apply_haar_filter(integral_images,16)
        haar16_r, haar16_g, haar16_b = haar16[0],haar16[1],haar16[2]

        if show_plot:
            # vertical prewitt plot 
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( vert_prewitt_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( vert_prewitt_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( vert_prewitt_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Vertical Prewitt of RGB image")
            plt.show()

            # horizontal prewitt plot
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( hori_prewitt_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( hori_prewitt_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( hori_prewitt_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Horizontal Prewitt of RGB image")
            plt.show()

            # laplace plot
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( laplace_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( laplace_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( laplace_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Laplacian of RGB image")
            plt.show()

            # gaussian plot
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( gauss_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( gauss_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( gauss_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Gaussian of RGB image")
            plt.show()


            # log of gaussian plot
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( l_gauss_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( l_gauss_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( l_gauss_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Log of Gaussian of RGB image")
            plt.show()

            # difference of gaussian plot
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16,4))
            plt.subplot(1,3,1), plt.imshow( d_gauss_r,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,2), plt.imshow( d_gauss_g,cmap="gray"), plt.axis("off")
            plt.subplot(1,3,3), plt.imshow( d_gauss_b,cmap="gray"), plt.axis("off")
            plt.suptitle("Difference of Gaussian of RGB image")
            plt.show()

            # LBP Red plot
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i, (img, label) in enumerate(zip(lbp_r, [4,8,16,24,32])):
                axes[i].imshow(img, cmap="gray")
                axes[i].axis('off')
                axes[i].set_title(label)  

            plt.suptitle("LBPs of Red image")
            plt.show()

            # LBP Green plot
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i, (img, label) in enumerate(zip(lbp_g, [4,8,16,24,32])):
                axes[i].imshow(img, cmap="gray")
                axes[i].axis('off')
                axes[i].set_title(label)  

            plt.suptitle("LBPs of Green image")
            plt.show()

            # LBP Blue plot
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i, (img, label) in enumerate(zip(lbp_b, [4,8,16,24,32])):
                axes[i].imshow(img, cmap="gray")
                axes[i].axis('off')
                axes[i].set_title(label)  

            plt.suptitle("LBPs of Blue image")
            plt.show()

            # Haar4 Filter plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i  in range(haar4.shape[0]):
                axes[i].imshow(haar4[i].astype(np.uint8),cmap="gray")
                axes[i].axis('off')

            plt.suptitle("Haar 4 of RGB image")
            plt.show()

            # Haar8 Filter plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i  in range(haar8.shape[0]):
                axes[i].imshow(haar8[i].astype(np.uint8),cmap="gray")
                axes[i].axis('off')

            plt.suptitle("Haar 8 of RGB image")
            plt.show()

            # Haar16 Filter plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i  in range(haar16.shape[0]):
                axes[i].imshow(haar16[i].astype(np.uint8),cmap="gray")
                axes[i].axis('off')

            plt.suptitle("Haar 16 of RGB image")
            plt.show()


        features = [
            vert_prewitt_r, hori_prewitt_r, 
            vert_prewitt_g, hori_prewitt_g, 
            vert_prewitt_b, hori_prewitt_b, 
            laplace_r, laplace_g, laplace_b,
            gauss_r, l_gauss_r, d_gauss_r,
            gauss_g, l_gauss_g, d_gauss_g,
            gauss_b, l_gauss_b, d_gauss_b,
            lbp_r[0],lbp_r[1],lbp_r[2],lbp_r[3],lbp_r[4],
            lbp_g[0],lbp_g[1],lbp_g[2],lbp_g[3],lbp_g[4],
            lbp_b[0],lbp_b[1],lbp_b[2],lbp_b[3],lbp_b[4],
            haar4[0],haar4[1],haar4[1],
            haar8[0],haar8[1],haar8[1],
            haar16[0],haar16[1],haar16[1],
            r, g, b,
            h, s, v,
        ]

        flattened_features = np.array([f[binary_mask].flatten() for f in features])
        # print(flattened_features[0].shape)

        return np.array(flattened_features)
    

    def dummy_test(self):
        # Mask,inverse and image (original in the lab1)

        image = cv2.imread("Images/image-35.jpg")
        mask = cv2.imread("Images/mask-35.png",cv2.IMREAD_GRAYSCALE)
        inverse_mask = 255-mask 
        class_inst = Stat_Classifier(image)

        # Validation features
        null = np.ones_like(mask)*255
        validation_img = cv2.imread("Images/image-83.jpg")


        # Get Features
        validation_features = class_inst.getFeatures(validation_img, null, show_plot=True)
        fg_features = class_inst.getFeatures(image, mask, show_plot=False)
        bg_features = class_inst.getFeatures(image, inverse_mask, show_plot=False)

        # Classify 
        verify_img = class_inst.classify(validation_features, fg_features, bg_features,mask,image)
        theta = 0.5
        thresholded_img = verify_img.copy() > theta
        plt.imshow(thresholded_img, cmap="gray"), plt.title("Validation image prediction")
        plt.show()