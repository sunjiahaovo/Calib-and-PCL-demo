import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import solve

def SVD_ICP(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]
    
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # print("AA = \n",AA)
    # print("BB = \n",BB)
    # rotation matrix
    H = AA.T @ BB
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
  
    # special reflection case
    if np.linalg.det(R) < 0:
       V[m-1,:] *= -1
       R = np.dot(V.T, U.T)
    # loss = np.linalg.norm(R@AA.T - BB.T, axis=0) 
    # print(loss)
    # translation
    t = centroid_B.T - R @ centroid_A.T
   
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    
    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


