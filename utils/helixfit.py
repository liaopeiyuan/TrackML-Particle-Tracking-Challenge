"""
fitting a set of points to a helix

Author: Peiyuan (Alexander) Liao

"""
import numpy as np
import scipy.linalg as la

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
    right_singular=-10*np.transpose(right_singular)
    S_vec=right_singular[:,np.argmin(singular),]
    S = np.zeros((4,4))
    
    S[0,0]=S_vec[0]
    S[0,1]=S_vec[1]
    S[0,2]=S_vec[2]
    S[0,3]=S_vec[3]
    
    S[1,0]=S_vec[1]
    S[1,1]=S_vec[4]
    S[1,2]=S_vec[5]
    S[1,3]=S_vec[6]

    S[2,0]=S_vec[2]
    S[2,1]=S_vec[5]
    S[2,2]=S_vec[7]
    S[2,3]=S_vec[8]

    S[3,0]=S_vec[3]
    S[3,1]=S_vec[6]
    S[3,2]=S_vec[8]
    S[3,3]=S_vec[9]

    det=np.linalg.det(S)

    if det==0:
        print(1)
    else:
        [eig_vals,eig_vecs]=np.linalg.eig(S)

    
    return eig_vals,eig_vecs

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

#print(helixfit(x,y,z))

#[a,b,c]=helixfit(x,y,z)
#print(a)
#print(b)
#print(c)