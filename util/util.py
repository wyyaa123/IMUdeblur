# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2023/07/04 21:22:17
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import sys
sys.path.append(f"F:\IMUdeblur")

import os
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import time
# from keras.utils import image_utils
from PIL import Image
import util.Image as _Image

def computeBlurfield(R: np.ndarray, 
                     K: np.ndarray, 
                     t: np.ndarray, 
                     height: np.int32, 
                     width: np.int32, 
                     part: np.int32 = 0, 
                     depth: np.float32 = np.float32('inf'),
                     n: np.ndarray = np.array[0, 0, 1]) -> tuple:
    Kinv = np.linalg.inv(K)

    xi, yi = np.meshgrid(range(width), range(height))
    xi = xi.astype(np.float_)
    yi = yi.astype(np.float_)

    Bx = np.zeros((height, width), dtype=np.float_)
    By = np.zeros((height, width), dtype=np.float_)

    for row in range(part, (part + 1) * height):
        x = xi[row, :]
        y = yi[row, :]
        z = np.ones(width, dtype=np.float_)
        X = np.vstack((x, y, z))

        # R1 = R[:, :, row]
        # R2 = R[:, :, row + nexp]
        # H = K.dot(R2).dot(R1.T).dot(Kinv)
        H = K @ (R + (1 / depth) * np.outer(t.T, n)) @ Kinv
        Xp = H @ X

        Xp[0, :] = Xp[0, :] / Xp[2, :]
        Xp[1, :] = Xp[1, :] / Xp[2, :]

        Bx[row, :] = np.around(Xp[0, :] - X[0, :]) # 矩阵元素取整, 四舍五入, Bx的元素是原图每一个像素点坐标经过在曝光时间内经过变换之后x方向上的变化
        By[row, :] = np.around(Xp[1, :] - X[1, :]) # 矩阵元素取整, 四舍五入, By的元素是原图每一个像素点坐标经过在曝光时间内经过变换之后y方向上的变化

    # Make sure that y-component is negative
    ypos = By > 0
    Bx[ypos] *= -1
    By[ypos] *= -1

    # Bx and By are saved as grayscale images. Cannot
    # have negative values so the value 128 is added.
    Bx = (Bx + 128).astype(np.uint8)
    By = (By + 128).astype(np.uint8)

    return Bx, By


# Visualize blur vectors and save the resulting image
# to the folder 'visualization'.
def plotBlurVectors(Bx, By, img, outpath, idx):
    Bx = Bx.astype(np.float_) - 128
    By = By.astype(np.float_) - 128

    # Display faded version of blurred image in the background
    img_bg = img.copy()
    img_bg = 0.5 * img_bg.astype(np.float_) + 100
    img_bg = np.clip(img_bg, 0, 255).astype(np.uint8)

    # Specify pixels for which to plot the blur vectors
    xsteps = 7
    ysteps = 5
    h, w = img.shape[:2]
    xvec = np.linspace(w / (xsteps + 1), w, xsteps, endpoint=False)
    yvec = np.linspace(h / (ysteps + 1), h, ysteps, endpoint=False)
    xvec = xvec.astype(np.int_)
    yvec = yvec.astype(np.int_)
    X, Y = np.meshgrid(xvec, yvec)
    X = X.ravel()
    Y = Y.ravel()

    # Overlay blur vectors on the background image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_bg)
    for x, y in zip(X, Y):
        u = np.array([x, x + Bx[y, x]])
        v = np.array([y, y + By[y, x]])
        u = x + (u - np.mean(u))  # Centering
        v = y + (v - np.mean(v))  # Centering
        plt.plot(u, v, 'r')

    plt.title('Image %d' % idx)
    plt.axis('off')

    # Save figure as image
    fname = '%04d.png' % (idx)
    plt.savefig(outpath + '/visualization/' + fname, bbox_inches='tight')
    plt.close()
    
    
def createOutputFolders(outpath):
    try:
        os.makedirs(outpath + '/blurred')
        os.makedirs(outpath + '/blurx')
        os.makedirs(outpath + '/blury')
        os.makedirs(outpath + '/visualization/')
    except FileExistsError:
        # Directory already exists
        pass
    
    
def writeImage(img, outpath, folder, idx):
    fname = '%04d.png' % idx
    img = Image.fromarray(img.astype(np.uint8))
    img.save(outpath + '/' + folder + '/' + fname)

    
if __name__ == '__main__':
    
    # =============================================================================================

    R_bi_to_w  = np.array([[ 0.78269262,  0.08492094, -0.61723999], # 1686660128.295778
                           [ 0.61881239,  0.00949855,  0.78599333],
                           [ 0.07258099, -0.99674607, -0.04509755]])

    R_bj_to_w  = np.array([[ 0.73215635,  0.08662814, -0.67606126], # 1686660128.3307908
                           [ 0.67718265,  0.02018225,  0.73595687],
                           [ 0.07737518, -0.99634553, -0.043873  ]])

    t_bi_to_w = np.array([-1.20150166, 0.29387971, 0.68290318])  # 1686660128.295778

    t_bj_to_w = np.array([-1.20639181, 0.25763614, 0.68240881])  # 1686660128.3307908

    # =============================================================================================
    
    
    K = np.array([[726.28741455078, 0, 354.6496887207],
                  [0, 726.28741455078, 186.46566772461],
                  [0, 0, 1]])

    T_cam_to_imu = np.array([[0.99972615, -0.02192887, -0.00817016, -0.02054695],
                             [0.02192867, 0.99975953, -0.00011467, 0.00232781],
                             [0.00817071, -0.00006452, 0.99996662, 0.03096576],
                             [0., 0., 0., 1.]])

    R_cam_to_imu = T_cam_to_imu[0:3, 0:3]
    t_cam_to_imu = T_cam_to_imu[0:3, 3:4].flatten()

    R_imu_to_cam = R_cam_to_imu.T
    t_imu_to_cam = -t_cam_to_imu

    Rij = R_imu_to_cam @ R_bj_to_w.T @ R_bi_to_w @ R_cam_to_imu
    tij = R_imu_to_cam @ R_bj_to_w.T @ R_bi_to_w @ t_cam_to_imu + R_imu_to_cam @ R_bj_to_w.T @ t_bi_to_w - R_imu_to_cam @ R_bj_to_w.T @ t_bj_to_w - R_imu_to_cam @ t_cam_to_imu
    
    img = cv.imread(f"../data/1686660128.338513.png", cv.IMREAD_COLOR)
    outpath = f"../data/"
    
    height, width = img.shape[:2]
    
    Bx, By = computeBlurfield(Rij, K, tij, height, width)

    createOutputFolders(outpath)
    writeImage(img, outpath, 'blurred/', idx=0)
    writeImage(Bx, outpath, 'blurx/', idx=0)
    writeImage(By, outpath, 'blury/', idx=0)
    
    plotBlurVectors(Bx, By, img, outpath, idx=0)
    
