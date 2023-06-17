'''
Descripttion: todo
Author: orCate
Date: 2023-05-19 15:18:54
LastEditors: orCate
LastEditTime: 2023-06-17 19:15:51
'''
import sys

sys.path.append("C:\\Users\\南九的橘猫\\Desktop\\IMUdeblur\\")
import numpy as np
import cv2 as cv
import include.Image as _Image
import matplotlib.pyplot as plt
import include.PSF as _PSF
import include.frame as _Frame
import time


def quaternion_to_Rotation(qu: np.ndarray) -> np.ndarray:
    """
    """
    R = np.zeros((3, 3), dtype=float)
    R[0, 0] = qu[0] ** 2 + qu[1] ** 2 - qu[2] ** 2 - qu[3] ** 2
    R[0, 1] = 2 * (qu[1] * qu[2] - qu[0] * qu[3])
    R[0, 2] = 2 * (qu[1] * qu[3] + qu[0] * qu[2])
    R[1, 0] = 2 * (qu[1] * qu[2] + qu[0] * qu[3])
    R[1, 1] = qu[0] ** 2 - qu[1] ** 2 + qu[2] ** 2 - qu[3] ** 2
    R[1, 2] = 2 * (qu[2] * qu[3] - qu[0] * qu[1])
    R[2, 0] = 2 * (qu[1] * qu[3] - qu[0] * qu[2])
    R[2, 1] = 2 * (qu[2] * qu[3] + qu[0] * qu[1])
    R[2, 2] = qu[0] ** 2 - qu[1] ** 2 - qu[2] ** 2 + qu[3] ** 2

    return R


def Roation_to_Euler(R: np.ndarray) -> np.ndarray:
    """
    """
    # 计算欧拉角
    # Pitch (绕x轴旋转) θ
    pitch = np.arcsin(R[1, 2])
    # heading (绕z轴旋转) ψ
    heading = np.arctan2(R[1, 0], R[1, 1])
    # Roll (绕y轴旋转) r
    roll = np.arctan2(-R[0, 2], R[2, 2])

    pitch_deg = np.degrees(pitch)
    heading_deg = np.degrees(heading)
    roll_deg = np.degrees(roll)

    return np.array([heading_deg, pitch_deg, roll_deg])


def deblur_byIMU(Rij: np.ndarray, tij: np.ndarray, raw_image: np.ndarray, depth_image: np.ndarray, K: np.ndarray,
                 N: int = 16,
                 overlap: int = 16) -> np.ndarray:
    """
    : param Rij: 两帧之间的旋转矩阵
    : param tij: 两帧之间的位移
    : param raw_image: 原模糊图像
    : param depth_image: 与模糊图像匹配的深度图
    : param K: 相机内参
    : param N: 原图要分割的块数
    : param overlap: 每一块的每一边要向外拓展的像素数
    """
    image_blocks = _Image.segment_nimage(raw_image, N, overlap, False)

    block_psfs, psfs_images = _Image.calcu_each_block_psf(False, image_blocks, depth_image, N, K, Rij, tij, overlap)

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for row in range(image_blocks.shape[0]):
        for col in range(image_blocks.shape[1]):
            # image_blocks[row, col] = _Image.winerFilter(image_blocks[row, col], block_psfs[row, col], 0.1, 0.01)
            image_blocks[row, col] = _Image.cls_filter(image_blocks[row, col], block_psfs[row, col], laplacian, 0.001)

    deblur_image = _Image.nimage_block_merge(image_blocks, N, overlap, False)
    psfs_merge = _Image.nimage_block_merge(psfs_images, N, overlap, False)
    return deblur_image


def make_motionblur(Rij: np.ndarray, tij: np.ndarray, raw_image: np.ndarray, depth_image: np.ndarray, K: np.ndarray,
                    N: int = 16,
                    overlap: int = 16) -> np.ndarray:
    """
    : param Rij: 两帧之间的旋转矩阵
    : param tij: 两帧之间的位移
    : param raw_image: 原模糊图像
    : param depth_image: 与模糊图像匹配的深度图
    : param K: 相机内参
    : param N: 原图要分割的块数
    : param overlap: 每一块的每一边要向外拓展的像素数
    """
    image_blocks = _Image.segment_nimage(raw_image, N, overlap, False)

    block_psfs, psfs_images = _Image.calcu_each_block_psf(False, image_blocks, depth_image, N, K, Rij, tij, overlap, True)

    motion_blur_image = _Image.nimage_block_merge(block_psfs, N, overlap, False)
    psfs_merge = _Image.nimage_block_merge(psfs_images, N, overlap, True)
    return motion_blur_image


if __name__ == '__main__':
    # ===========================================================================================

    # R_i = np.array([[-0.91846082, -0.0865836, 0.38610487], # 相差10ms
    #                 [-0.38811699, 0.00707313, -0.92166108],
    #                 [0.07706422, -0.99629174, -0.04009807]])  # 1686660083.491411

    # R_j = np.array([[-0.91395729, -0.08704112, 0.39656443],
    #                 [-0.39838847, 0.0039503, -0.91729409],
    #                 [0.0782696, -0.9962759, -0.03828357]])  # 1686660083.503916

    # t_i = np.array([3.93132203, 1.20911013, 0.71370362])  # 1686660083.491411
    # t_j = np.array([3.94323513, 1.21811144, 0.71319248])  # 1686660083.503916

    # ===========================================================================================

    R_i = np.array([[-0.28162419, 0.01531572, -0.95983503],
                    [0.95927558, 0.04217897, -0.28078701],
                    [0.03616939, -0.99940802, -0.02655959]])  # 1686660103.5867586

    R_j = np.array([[-0.26114005, 0.01669909, -0.96544284],
                    [0.96490992, 0.04196719, -0.26027],
                    [0.03616066, -0.99925611, -0.02706495]])  # 1686660103.5992334

    t_i = np.array([1.47639905, 0.44433637, 0.57882645])  # 1686660103.5867586
    t_j = np.array([1.4655684, 0.45022566, 0.57860039])  # 1686660103.5992334

    # ===========================================================================================

    # R_i = np.array([[-0.99256143, -0.07926868,  0.09297691], # 1686660083.461399
    #                 [-0.09579978,  0.03257912, -0.99492081],
    #                 [ 0.07583292, -0.99637417, -0.03992858]])

    # R_j = np.array([[-0.98929817, -0.0806827,   0.12190194] # 1686660083.4964125
    #                 [-0.12497486,  0.03418366, -0.99161152]
    #                 [ 0.07583578, -0.99619397, -0.04389938]])

    # t_i = np.array([3.2050875,  0.82298972, 0.49929595])  # 1686660083.461399

    # t_j = np.array([3.24369922, 0.83751518, 0.49717086])  # 1686660083.4964125

    # ===========================================================================================

    # R_i = np.array([[ 0.94784908,  0.12555788, -0.2933164 ], # 1686660095.7390637
    #                 [ 0.30375458, -0.0737625,   0.95000496],
    #                 [ 0.09763425, -0.98945009, -0.10804277]])

    # R_j = np.array([[ 0.93427278,  0.12824434, -0.33294694], # 1686660095.7715793
    #                 [ 0.34338161, -0.06973711,  0.9366919 ],
    #                 [ 0.09689864, -0.98937153, -0.10918117]])

    # t_i = np.array([0.72279245, 1.68415999, 0.46360742])  # 1686660095.7390637

    # t_j = np.array([0.70494404, 1.6525445,  0.46339367])  # 1686660095.7715793

    # =============================================================================================

    # R_i = np.array([[-0.44073751,  0.00646143, -0.89793164], # 1686660103.5593135
    #                 [ 0.89694646,  0.05055928, -0.43989013],
    #                 [ 0.04254428, -0.99898675, -0.02807089]])

    # R_j = np.array([[-0.38246239,  0.00986317, -0.92413358], # 1686660103.591749
    #                 [ 0.92342229,  0.04473489, -0.38169057],
    #                 [ 0.03756886, -0.99914921, -0.02621207]])

    # t_i = np.array([1.5511742, 0.45377332, 0.5593354])  # 1686660103.5593135

    # t_j = np.array([1.52029517, 0.46124946, 0.56066473])  # 1686660103.591749

    # =============================================================================================

    # R_i = np.array([[ 0.78269262,  0.08492094, -0.61723999], # 1686660128.295778
    #                 [ 0.61881239,  0.00949855,  0.78599333],
    #                 [ 0.07258099, -0.99674607, -0.04509755]])

    # R_j = np.array([[ 0.73215635,  0.08662814, -0.67606126], # 1686660128.3307908
    #                 [ 0.67718265,  0.02018225,  0.73595687],
    #                 [ 0.07737518, -0.99634553, -0.043873  ]])

    # t_i = np.array([-1.20150166, 0.29387971, 0.68290318])  # 1686660128.295778

    # t_j = np.array([-1.20639181, 0.25763614, 0.68240881])  # 1686660128.3307908

    # =============================================================================================

    # blur_image = cv.imread("./1686660103.600574.png", cv.IMREAD_GRAYSCALE)
    # depth_image = np.load("./1686660103.600574.npy")
    
    # scaled_image = cv.normalize(depth_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # # test_image = depth_image / 256
    # plt.imshow(depth_image, cmap="gray")
    # plt.show()
    sharp_image = cv.imread("./test1/1686713423.894630.png", cv.IMREAD_GRAYSCALE)

    # cv.imshow("rsize", sharp_image)
    # cv.waitKey()

    depth_image = np.load("./test1/1686713423.894630.npy")

    K = np.array([[591.9406904278699, 0, 322.04277865756234],
                  [0, 591.2925794176291, 248.3881726957238],
                  [0, 0, 1]])

    # T_imu_to_cam = np.array([[0.99972615, -0.02192887, -0.00817016, -0.02054695],
    #                          [0.02192867, 0.99975953, -0.00011467, 0.00232781],
    #                          [0.00817071, -0.00006452, 0.99996662, 0.03096576],
    #                          [0., 0., 0., 1.]])

    # R_imu_to_cam = T_imu_to_cam[0:3, 0:3]
    # t_imu_to_cam = T_imu_to_cam[0:3, 3:4].flatten()

    # Rij = R_j @ R_i.T  # 先计算出在IMU坐标系下的旋转矩阵和平移
    # tij = t_j - Rij @ t_i
    # tij = R_imu_to_cam.T @ Rij @ t_imu_to_cam + R_imu_to_cam.T @ tij - R_imu_to_cam.T @ t_imu_to_cam  # 再计算在相机系下的旋转和平移
    # Rij = R_imu_to_cam.T @ Rij @ R_imu_to_cam
    
    Rij = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    tij = np.array([0.1, 0.1, 0])

    print("Rij is", Rij)
    print("tij is", tij)

    beg_time = time.time()
    # deblur_image = deblur_byIMU(Rij, tij, blur_image, depth_image, K)
    blur_image = make_motionblur(Rij, tij, sharp_image, depth_image, K)
    print("elasped {0:.6f} seconds".format(time.time() - beg_time))
    # # 显示灰度图像
    # # plt.imshow(deblur_image, cmap='gray')
    # # plt.axis('off')  # 关闭坐标轴
    # # plt.show()

    # cv.namedWindow("deblur", cv.WINDOW_NORMAL)
    # cv.imshow("deblur", deblur_image / 256)
    # cv.waitKey()
    cv.namedWindow("blur", cv.WINDOW_NORMAL)
    cv.imshow("blur", blur_image / 256)
    cv.waitKey()
