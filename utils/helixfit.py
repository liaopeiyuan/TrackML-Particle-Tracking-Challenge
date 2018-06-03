"""
fitting a set of points to a helix

Author: Peiyuan (Alexander) Liao

"""
import numpy as np

def helixfit(x,y,z):
    
    # 1. Estimating the degree of collinearity of the data
    xmean=np.mean(x)
    ymean=np.mean(y)
    zmean=np.mean(z)
    

    X = np.zeros((np.prod(x.shape),3))

    for i in range(np.prod(x.shape)):
        X[i-1,0]=x[i-1]-xmean
        X[i-1,1]=y[i-1]-ymean
        X[i-1,2]=z[i-1]-zmean

    [U,singular,V]=np.linalg.svd(X)
    sigma0=singular[2]*np.random.rand(1)

    # 2. Fitting the axis and the radius of the helix
    Z = np.ones((np.prod(x.shape),10))

    for i in range(np.prod(x.shape)):
        cur_x=x[i-1]
        cur_y=y[i-1]
        cur_z=z[i-1]
        Z[i-1,0]=cur_x**2
        Z[i-1,1]=2*cur_x*cur_y
        Z[i-1,2]=2*cur_x*cur_z
        Z[i-1,3]=2*cur_x
        Z[i-1,4]=cur_y**2
        Z[i-1,5]=2*cur_y*cur_z
        Z[i-1,6]=2*cur_y
        Z[i-1,7]=cur_z**2
        Z[i-1,8]=2*cur_z

    #ZtZ=np.dot(Z,np.transpose(Z))
    [_,singular,right_singular]=np.linalg.svd(Z)
    S=np.amin(singular)

    return singular,right_singular

x=np.array([62,82,93,94,65,12,48,77,85,89])
y=np.array([397,347,288,266,163,102,138,187,209,316])
z=np.array([103,107,120,128,169,198,180,157,149,112])
#x=np.random.rand(5,1)
#print(x)
#y=np.random.rand(5,1)
#z=np.random.rand(5,1)
[a,b]=helixfit(x,y,z)
print(a)
print(b)
#[a,b,c]=helixfit(x,y,z)
#print(a)
#print(b)
#print(c)