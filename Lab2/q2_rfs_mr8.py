import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from q1_filters import LoG,GaussFilter

def create_gaussian_filter(theta, sigma_x, sigma_y, size, filter_type='edge'):
    # Create a grid of (x, y) coordinates
    x = np.linspace(-size//2+1, size//2, size)
    y = np.linspace(-size//2+1, size//2, size)
    x, y = np.meshgrid(x, y)
    # Rotate the coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # Calculate the Gaussian function f(xrot,sigmax)*f(yrot,sigmay)
    fx = np.exp(-0.5 * (x_rot**2 / sigma_x**2))/(np.sqrt(2 * np.pi) * sigma_x)
    fy = np.exp(-0.5 * (y_rot**2 / sigma_y**2))/(np.sqrt(2 * np.pi) * sigma_y)
    if filter_type == 'edge':
        # First derivative (edge)
        #x'
        dG_dx = fy*fx*(-x_rot/sigma_x**2)
        #y'
        dG_dy = fx*fy*(-y_rot/sigma_y**2)
        return dG_dx, dG_dy
    elif filter_type == 'bar':
        # Second derivative (bar)
        #x'
        d2G_dx2 = fy*fx*((x_rot-sigma_x**2)/sigma_x**4)
        #y'
        d2G_dy2 = fx*fy*((y_rot-sigma_y**2)/sigma_y**4)
        return d2G_dx2, d2G_dy2
    else:
        raise ValueError("Unknown filter type. Use 'edge' or 'bar'.")
    
def construct_rfs(debug: bool = False):
    """
    Args:
        debug (bool): Whether you want the function to plot the filters.

    Returns:
        array (2d): The 36 filters in a (6x6) array the first 3 rows are edge, the last 3 are bar.
    """
    sigma_x_sigma_y = np.array([(3,1),(6,2),(12,4)])
    thetas = np.array([0, 1/6*np.pi, 2/6*np.pi, 3/6*np.pi, 4/6*np.pi, 5/6*np.pi])

    size = (49, 49)

    rfs_edge = np.zeros((sigma_x_sigma_y.shape[0], thetas.shape[0], size[0], size[1]))
    rfs_bar = np.zeros((sigma_x_sigma_y.shape[0], thetas.shape[0], size[0], size[1]))

    for sigma_ind in range(sigma_x_sigma_y.shape[0]):
        for theta_ind in range(thetas.shape[0]):
            sigma = sigma_x_sigma_y[sigma_ind]
            theta = thetas[theta_ind]

            gaussian_edge = create_gaussian_filter(theta, sigma[0], sigma[1], size[0], 'edge')
            rfs_edge[sigma_ind, theta_ind] = gaussian_edge[1]
            gaussian_bar = create_gaussian_filter(theta, int(sigma[0]), sigma[1], size[0], 'bar')
            rfs_bar[sigma_ind, theta_ind] = gaussian_bar[1]

    # Combine rfs_edge and rfs_bar by concatenating along the theta axis
    rfs_combined = np.concatenate((rfs_edge, rfs_bar), axis=0)
    
    print(rfs_combined.shape)

    def plot_filters(filters, title, size=(49, 49)):
        rows, cols = filters.shape[:2]
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)
        
        for i in range(rows):
            for j in range(cols):
                ax = axes[i, j]
                ax.imshow(filters[i, j], cmap='inferno')
                ax.axis('off')
        
        plt.show()

    if debug:  
        plot_filters(rfs_combined, title="Combined Edge and Bar Filters (Y component)")
    
    return rfs_combined

def apply_rfs_filter_scipy(image, rfs_filters):
    """
    Applies the filters given and returns the results.

    Args:
        image (2d): the image.
        rfs_filters (2d): a 6 by 6 matrix (1-3 edge, 4-6 bar).

    Returns:
        Array (3d): An array that is (image.shape[0] x image.shape[1] x 8).
    """
    max_responses = np.zeros((image.shape[0], image.shape[1], rfs_filters.shape[0] +2)) # plus 2 for the log and gauss

    for sigma_ind in range(rfs_filters.shape[0]):
        # Edge filters
        responses = []
        for theta_ind in range(rfs_filters.shape[1]):
            filter = rfs_filters[sigma_ind, theta_ind]
            response = convolve(image, filter)
            responses.append(response)
        
        max_responses[:, :, sigma_ind] = np.max(responses, axis=0)
    # now apply log and gauss and add them to the responses at the end of np array
    sigma = 10**0.5

    log_response = convolve(image,LoG(49, sigma))
    gauss_response = convolve(image,GaussFilter(49, sigma))
    max_responses[:,:,max_responses.shape[2]-2] = log_response
    max_responses[:,:,max_responses.shape[2]-1] = gauss_response
    
    return max_responses