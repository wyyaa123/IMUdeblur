'''
Descripttion: todo
Author: orCate
Date: 2023-05-19 15:18:54
LastEditors: orCate
LastEditTime: 2023-06-07 10:00:05
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


def deblur_byIMU(Rij: np.ndarray, tij: np.ndarray, raw_image: cv.Mat, depth_image: cv.Mat, K: np.ndarray, N: int = 16,
                 overlap: int = 16) -> cv.Mat:
    image_blocks = _Image.segment_nimage(raw_image, N, overlap, False)
    block_psfs = _Image.calcu_each_block_psf(False, image_blocks, depth_image, N, K, Rij, tij, overlap)

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for row in range(image_blocks.shape[0]):
        for col in range(image_blocks.shape[1]):
            image_blocks[row, col] = _Image.cls_filter(image_blocks[row, col], block_psfs[row, col], laplacian, 0.01)

    deblur_image = _Image.nimage_block_merge(image_blocks, N, overlap, False)
    return deblur_image


if __name__ == '__main__':
    # R_i = np.array([[0.02964806, 0.99948285, 0.01248415], # 1685622900.402839
    #                [-0.01521195, 0.01293936, -0.99980098],
    #                [-0.99944506, 0.02945224, 0.01558771]])

    # R_j = np.array([[0.03134772, 0.99933626, 0.01863067],
    #                [-0.01285934, 0.01904152, -0.99973737],
    #                [-0.99942719, 0.03109986, 0.0134477 ]])

    # t_i = np.array([0.19998459, -0.21885501, 0.01411131])
    # t_j = np.array([0.21272098, -0.22780293, 0.01434683])

    # R_i = np.array([[0.01656386, 0.98862055, 0.1496011], # 1685622902.8751712
    #                [-0.07176211, 0.1504112, -0.98602845],
    #                [-0.99729698, 0.00559668, 0.07343595]])

    # R_j = np.array([[0.02266376, 0.9863049, 0.16347437],
    #                [-0.07180252, 0.16469991, -0.98374448],
    #                [-0.99717883, 0.01055729, 0.07455059]])

    # t_i = np.array([1.08159584, -0.22992354, -0.03657288])
    # t_j = np.array([1.13673438, -0.23298174, -0.03948713])

    R_i = np.array([[0.02123373, 0.95523879, 0.29514681],  # 1685622903.20405
                    [-0.04231862, 0.29580751, -0.9543326],
                    [-0.99890036, 0.00777367, 0.04670446]])

    R_j = np.array([[0.02433315, 0.95344428, 0.30069328],  # 1685622903.2540727
                    [-0.06882506, 0.30166633, -0.95096008],
                    [-0.9973643, 0.00244454, 0.072959]])

    t_i = np.array([1.50853111, -0.18809431, -0.02557929])  # 1685622903.20405
    t_j = np.array([1.56890657, -0.17458322, -0.02635776])  # 1685622903.2540727

    K = np.array([[407.76044960599006, 0, 309.58075680219565],
                  [0, 406.5997999128962, 193.42444301176783],
                  [0, 0, 1]])

    blur_image = cv.imread("./blur_image.png", cv.IMREAD_GRAYSCALE)
    depth_image = cv.imread("./depth_image.png", cv.IMREAD_ANYDEPTH)

    # depth_image = np.load("./1686021820_443643.npy")

    T_imu_to_cam = np.array([[0.00647089, -0.99997884, -0.00066787, 0.05393405],
                             [0.99997256, 0.00646844, 0.00361214, 0.01340939],
                             [-0.00360775, -0.00069122, 0.99999325, -0.00295812],
                             [0., 0., 0., 1.]])

    R_imu_to_cam = T_imu_to_cam[0:3, 0:3]
    t_imu_to_cam = T_imu_to_cam[0:3, 3:4].flatten()

    # beg_time = time.time()
    Rij = R_j @ np.linalg.inv(R_i)
    tij = t_j - Rij @ t_i
    tij = R_imu_to_cam.T @ Rij @ t_imu_to_cam + R_imu_to_cam.T @ tij - R_imu_to_cam @ t_imu_to_cam
    Rij = R_imu_to_cam.T @ Rij @ R_imu_to_cam

    deblur_image = deblur_byIMU(Rij, tij, blur_image, depth_image, K)
    # print("elasped {0:.6f} seconds".format(time.time() - beg_time))
    # 显示灰度图像
    cv.imshow("deblur_image", deblur_image / 255)
    cv.waitKey()
