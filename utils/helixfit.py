"""
fitting a set of points to a helix

Author: Peiyuan (Alexander) Liao

"""
import numpy as np

def helixfit(x,y,z):
    
    # Arithmetic mean of points
    xmean=np.mean(x)
    ymean=np.mean(y)
    zmean=np.mean(z)

    X = np.zeros(np.prod(x.shape),3)
    for i in range(np.prod(x.shape))
        X(i,1)=x(i)-xmean
        X(i,2)=y(i)-ymean
        X(i,3)=z(i)-zmean


    return X
