# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2023/09/14 16:54:17
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import numpy as np
import time

# https://github.com/jannemus/DeepGyro/blob/master/preprocess/utils.py
def computeBlurfield(R, K, height, width):

    Kinv = np.linalg.inv(K)
    
    xi, yi = np.meshgrid(range(width),range(height))
    xi = xi.astype(np.float_)
    yi = yi.astype(np.float_)
    
    Bx = np.zeros((height,width),dtype=np.float_)
    By = np.zeros((height,width),dtype=np.float_)
    
    for row in range(height):
    
        x = xi[row, :]
        y = yi[row, :]
        z = np.ones(width, dtype=np.float_)
        X = np.vstack((x,y,z))
        
        H = K.dot(R).dot(Kinv)
        Xp = H.dot(X)

        Xp[0,:] = Xp[0,:]/Xp[2,:] # 归一化
        Xp[1,:] = Xp[1,:]/Xp[2,:]

        Bx[row,:] = np.around(Xp[0,:] - X[0,:])
        By[row,:] = np.around(Xp[1,:] - X[1,:])
        
    # Make sure that y-component is negative
    ypos = By > 0
    Bx[ypos] = -1 * Bx[ypos]
    By[ypos] = -1 * By[ypos]
    
    # Bx and By are saved as grayscale images. Cannot
    # have negative values so the value 128 is added.
    Bx = (Bx + 128).astype(np.uint8)
    By = (By + 128).astype(np.uint8)
    
    return Bx, By

if __name__ == "__main__":


    # R = np.array([[1, 0, 0], 
    #               [0, 1, 0], 
    #               [0, 0, 1]])

    R = np.array([[0.9986, -0.05234, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转3°
                  [0.05234, 0.9986, 0],
                  [0, 0, 1]])
    
    K = np.array([[319.630096435547, 0, 387.966674804688],
                  [0, 234.015075683594, 387.966674804688],
                  [0, 0, 1]])

    beg = time.time()
    Bx, By = computeBlurfield(R, K, 480, 640)
    print (time.time() - beg)
