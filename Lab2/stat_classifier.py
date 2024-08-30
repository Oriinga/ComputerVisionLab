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
    def classify(self,image_features,fg_features,bg_features):
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
        
        
        reshaped_features = image_features.T
        
        
        ####### vector of predictions #######
        probabilities = self.foreground_given_pixel(reshaped_features, fg_mean_vector, fg_cov_matrix, bg_mean_vector, bg_cov_matrix)
        
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

        features = [
            vert_prewitt_r, hori_prewitt_r,
            vert_prewitt_g, hori_prewitt_g,
            vert_prewitt_b, hori_prewitt_b,
            laplace_r, laplace_g, laplace_b,
            r, g, b,
            h, s, v
        ]

        flattened_features = np.array([f[binary_mask].flatten() for f in features])
        # print(flattened_features[0].shape)

        return np.array(flattened_features)
    


image = cv2.imread("Images/image-35.jpg")

plt.imshow(image)
plt.show()

class_inst = Stat_Classifier(image)
mask = cv2.imread("Images/mask-35.png",cv2.IMREAD_GRAYSCALE)
features = class_inst.getFeatures(image,mask,True)

class_inst.classify(cv2.imread("Images/image-83.jpg"),)