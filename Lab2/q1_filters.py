import numpy as np


def GaussFilter(size, sigma):
    values =np.arange(-(size//2), size//2 + 1)
    x, y = np.meshgrid(values, values)
    g = (1/(2*np.pi * sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    #normalisng it - sometimes the sum is not perfectly 1...
    g /= np.sum(g)
    # print(np.sum(g))
    return g

def DoG(size, sigma, K):
    values =np.arange(-(size//2), size//2 + 1)
    x, y = np.meshgrid(values, values)
    
    g = (1/(2 * np.pi * sigma **2)) * \
        np.exp(-(x**2 + y ** 2)/(2 * sigma **2)) - \
        (1/ (2 * np.pi * K**2 *sigma**2)) * \
        np.exp(-(x**2 + y ** 2)/(2 * K**2 * sigma **2))

    return g 
def LoG(size, sigma):
    values =np.arange(-(size//2), size//2 + 1)
    x, y = np.meshgrid(values, values)
    
    g = -((1)/(np.pi * sigma ** 4 )) * \
        (1- (x**2 + y**2)/(2* sigma **2)) * \
        np.exp(-(x**2 + y**2)/(2*sigma**2))
    

    return g