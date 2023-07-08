# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/07 20:13:21
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import sys
from keras.utils import image_utils
# from keras.preprocessing.image import array_to_img
from models import modelsClass
from matplotlib import pyplot as plt

sys.path.append(f"F:\IMUdeblur")

import util.util as _util
import util.Image as _Image
import numpy as np
import cv2 as cv
import time

if __name__ == '__main__':

    # 获得原始数据: R t depth color_img out_path
    # =============================================================================================

    R_bi_to_w = np.array([[0.78269262, 0.08492094, -0.61723999],  # 1686660128.295778
                          [0.61881239, 0.00949855, 0.78599333],
                          [0.07258099, -0.99674607, -0.04509755]])

    R_bj_to_w = np.array([[0.73215635, 0.08662814, -0.67606126],  # 1686660128.3307908
                          [0.67718265, 0.02018225, 0.73595687],
                          [0.07737518, -0.99634553, -0.043873]])

    t_bi_to_w = np.array([-1.20150166, 0.29387971, 0.68290318])  # 1686660128.295778

    t_bj_to_w = np.array([-1.20639181, 0.25763614, 0.68240881])  # 1686660128.3307908

    blurred_img = cv.imread(f"../data/1686660128.338513.png", cv.IMREAD_UNCHANGED)
    depth_img = np.load(f"../data/1686660128.338513.npy")
    height, width = blurred_img.shape[:2]

    outpath = f"../output"
    _util.createOutputFolders(outpath)

    bgr = False
    if blurred_img.ndim != 2:  # 如果不是灰度图
        bgr = True

    # =============================================================================================

    # 计算两帧之间的位姿变化
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

    # =============================================================================================

    # 将图像分块计算每一块的Bx, By, 再将每一个Bx, By拼合
    # =============================================================================================

    N = 16  # NxN块
    overlap = 0  # 重叠的像素点
    view_blocks = False
    bx_blocks = np.zeros((N, N), dtype=np.ndarray)
    by_blocks = np.zeros((N, N), dtype=np.ndarray)
    height_base, width_base = height // N, width // N

    models = modelsClass(height, width)
    model = models.getDeepGyro()
    model.load_weights("../checkpoints/DeepGyro.hdf5")

    # 如果你想看分块后的结果 view_blocks = True 

    if bgr and view_blocks:
        color_blocks = np.zeros((N, N), dtype=np.ndarray)
        b_blocks = _Image.segment_nimage(blurred_img[:, :, 0], N, overlap, False)
        g_blocks = _Image.segment_nimage(blurred_img[:, :, 1], N, overlap, False)
        r_blocks = _Image.segment_nimage(blurred_img[:, :, 2], N, overlap, False)

        fig, axes = plt.subplots(nrows=N, ncols=N)
        for row in range(N):
            for col in range(N):
                color_blocks[row, col] = np.dstack((b_blocks[row, col], g_blocks[row, col], r_blocks[row, col]))
                axes[row, col].imshow(color_blocks[row, col], cmap="gray", vmin=0, vmax=255)
                # axes[i, j].set_title("block{0}{1}".format(i, j))
                axes[row, col].axis('off')

        fig.suptitle(f"image segmentation {N} x {N}")

        # 调整子图的布局
        plt.tight_layout()

        # 显示图像界面
        plt.show()

    elif not bgr and view_blocks:
        gray_blocks = _Image.segment_nimage(blurred_img[:, :, 0], N, overlap, True)

    beg_time = time.time()
    # 反之, 如果你想保证运算速度 view_blocks = False
    for row in range(0, N):
        for col in range(0, N):
            blocki_y, blocki_x = height_base * (2 * row + 1) / 2, width_base * (2 * col + 1) / 2
            depth = (_Image.get_avedepth(depth_img, blocki_x, blocki_y, N) if not depth_img[
                int(blocki_y), int(blocki_x)] else depth_img[int(blocki_y), int(blocki_x)]) * 1e-3
            bx_blocks[row, col], by_blocks[row, col] = _util.computeBlurfield(Rij, K, tij, height_base, width_base,
                                                                              (row, col), depth)

    print()

    Bx = _Image.nimage_block_merge(bx_blocks, N, overlap, False)
    By = _Image.nimage_block_merge(by_blocks, N, overlap, False)

    # _util.writeImage(raw_img, outpath, 'blurred/', idx=0)
    # _util.writeImage(Bx, outpath, 'blurx/', idx=0)
    # _util.writeImage(By, outpath, 'blury/', idx=0)

    blurred_np = (1. / 255) * blurred_img
    blurx_np = (1. / 255) * Bx
    blury_np = (1. / 255) * By

    b = np.reshape(blurred_np, [1, height, width, 3])
    bx = np.reshape(blurx_np, [1, height, width, 1])
    by = np.reshape(blury_np, [1, height, width, 1])
    x = [b, bx, by]

    # beg_time = time.time()
    prediction = model.predict(x, batch_size=5, verbose=0, steps=None)
    prediction = prediction[0, :, :, :]

    # deblurred_img = array_to_img(prediction)
    deblurred_img = image_utils.array_to_img(prediction)
    end_time = time.time()
    print("elasped {0:0.5}".format(end_time - beg_time))
    print("psnr is", _Image.get_psnr(blurred_img, np.array(deblurred_img)))
    plt.imshow(deblurred_img)
    plt.title("Image")
    plt.axis("off")  # 关闭坐标轴显示
    plt.show()

