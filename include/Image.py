#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023-05-28 16:31
# @Author : orCate

import sys

from numpy import ndarray
import scipy

sys.path.append("C:\\Users\\南九的橘猫\\Desktop\\IMUdeblur\\")
import numpy as np
import matplotlib.pyplot as plt
from functools import singledispatch
import cv2 as cv
import math
import include.PSF as PSF


def image_fft(gray, vis: bool = None) -> None:
    """
    计算图像的频谱
    :param gray: 灰度图
    :param vis: 是否显示
    """
    if gray.ndim != 2:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)
    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (_, ax) = plt.subplots(1, 2, )
        ax[0].imshow(gray, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()


def segment_nimage(gray: np.ndarray, n: int, overlap: int, vis: bool = None) -> np.ndarray:
    """
    分割图像,分成NxN份
    :param gray:灰度图
    :param n: NxN份
    :param overlop: 重叠的像素大小
    :param vis: 是否显示分割后的图像
    :return image_blocks: 返回分割的图像块数组
    """
    if gray.ndim != 2:  # 如果不是灰度图
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    image_blocks = np.empty((n, n), dtype=object)

    height, width = gray.shape
    assert height // n == height / n and width // n == width / n
    height_base = height // n
    width_base = width // n

    for i in range(n):  # row
        for j in range(n):  # col
            up_pixel = overlap if i else 0
            down_pixel = overlap if i + 1 and i + 1 != n else 0
            left_pixel = overlap if j else 0
            right_pixel = overlap if j + 1 and j + 1 != n else 0
            image_blocks[i, j] = gray[i * height_base - up_pixel: (i + 1) * height_base + down_pixel,
                                 j * width_base - left_pixel: (j + 1) * width_base + right_pixel]
            # print("image_blocks[{0}][{1}] is ({2},{3})".
            #       format(i, j, image_blocks[i, j].shape[0], image_blocks[i, j].shape[1]))

    if vis:
        # 创建一个包含多个子图的图像界面
        for i in range(image_blocks.shape[0]):
            for j in range(image_blocks.shape[1]):
                print("第{0}行, 第{1}列, 大小:{2}".format(i, j, image_blocks[i, j].shape))

        _, axes = plt.subplots(nrows=n, ncols=n)
        for i in range(n):  # row
            for j in range(n):  # col
                axes[i, j].imshow(image_blocks[i, j], cmap="gray", vmin=0, vmax=255)
                # axes[i, j].set_title("block{0}{1}".format(i, j))
                axes[i, j].axis('off')

        # print("expired {0:.12f} seconds".format(time.time() - beg_time))

        # 调整子图的布局
        plt.tight_layout()

        # 显示图像界面
        plt.show()
        # print("expired {0:.12f} seconds".format(time.time() - beg_time))

    return image_blocks


def swap_diag_blocks(gray):
    """
    对角块互换
    :param gray: 灰度图
    :return merged_mat: 互换之后的图像
    """
    h, w = gray.shape
    h1 = math.ceil(h / 2)
    w1 = math.ceil(w / 2)
    mat1 = gray[0:h1, 0:w1]
    mat2 = gray[0:h1, w1:w]
    mat3 = gray[h1:h, 0:w1]
    mat4 = gray[h1:h, w1:w]

    temp = np.copy(mat1)
    mat1 = np.copy(mat4)
    mat4 = np.copy(temp)
    temp = np.copy(mat2)
    mat2 = np.copy(mat3)
    mat3 = np.copy(temp)

    merged_mat = np.vstack((np.hstack((mat1, mat2)), np.hstack((mat3, mat4))))
    return merged_mat


def calWeight(d, k):
    """
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return y: 权重
    """

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, block_shape, cnt: int, left_right=None) -> np.ndarray:
    """
    边界按权重融合的两幅图像
    :param img1: 第一幅要融合的图像
    :param img2: 第二幅要融合的图像
    :param overlap: 重叠的像素大小
    :param block_shape: 块的形状, 通常是要融合的图像序列中的第一幅图像
    :param cnt: 
    :param left_right: 是左右融合还是上下融合
    :return img_new: 融合出的新图像
    """
    w = calWeight(overlap, 0.05)  # k=5 这里是超参
    if left_right:  # 左右融合
        row, col = block_shape[0], (cnt + 1) * (block_shape[1] - overlap // 2) + overlap // 2
        # img_new = np.zeros((row, 2 * col - overlap))
        img_new = np.zeros((row, col))
        img_new[:, :img1.shape[1]] = img1
        w_expand = np.tile(w, (row, 1))  # 权重扩增
        img_new[:, img1.shape[1] - overlap:img1.shape[1]] = \
            (1 - w_expand) * img1[:, img1.shape[1] - overlap:img1.shape[1]] + \
            w_expand * img2[:, :overlap]
        img_new[:, img1.shape[1]:img1.shape[1] + img2.shape[1] - overlap] = img2[:, overlap:]
    else:  # 上下融合
        row, col = (cnt + 1) * (block_shape[0] - overlap // 2) + overlap // 2, block_shape[1]
        img_new = np.zeros((row, col))
        img_new[:img1.shape[0], :] = img1
        w = np.reshape(w, (overlap, 1))
        w_expand = np.tile(w, (1, col))
        img_new[img1.shape[0] - overlap:img1.shape[0], :] = \
            (1 - w_expand) * img1[img1.shape[0] - overlap:img1.shape[0], :] + w_expand * img2[:overlap, :]
        img_new[img1.shape[0]:img1.shape[0] + img2.shape[0] - overlap, :] = img2[overlap:, :]
    return img_new


def nimage_block_merge(image_blocks: np.ndarray, n: int, overlap: int, vis: bool = None) -> np.ndarray:
    """
    将(NxN)个图像块合并
    :param image_blocks:图像块数组
    :param n: 有NxN块
    :param overlap: 重叠区域
    :param vis: 是否显示
    :return raw_image: 返回合并后的图像
    """
    overlap *= 2
    # height_base, width_base = gray_block.shape[0], gray_block.shape[1]
    # raw_image = np.zeros((height_base * n, width_base * n))

    temp_imgblock = np.empty(n, object)

    for i in range(n):
        cnt: int = 1
        temp_img = image_blocks[i, 0]
        for j in range(1, n):
            gray_block = image_blocks[i, j]
            if gray_block.ndim != 2:
                gray_block = cv.cvtColor(gray_block, cv.COLOR_BGR2GRAY)
            # raw_image[i * height_base: (i + 1) * height_base, j * width_base: (j + 1) * width_base] = gray_block
            # temp_img = imgFusion(temp_img, image_blocks[i, j], overlop, True)
            # print(i, j)
            # print("temp_img.shape is", temp_img.shape)
            temp_img = imgFusion(temp_img, gray_block, overlap, image_blocks[i, 0].shape, cnt, True)
            # print("gray_block.shape is", gray_block.shape)
            # print("temp_img.shape is", temp_img.shape)
            # os.system("cls")
            # cv.imshow("temp_img", temp_img / 256)
            # cv.waitKey()
            temp_imgblock[i] = temp_img
            cnt += 1

    cnt = 1
    temp_img = temp_imgblock[0]
    for i in range(1, n):
        temp_img = imgFusion(temp_img, temp_imgblock[i], overlap, temp_imgblock[0].shape, cnt, False)
        cnt += 1

    raw_image = temp_img[:temp_img.shape[0] - overlap // 2, :temp_img.shape[1] - overlap // 2]

    if vis:
        # plt.imshow(raw_image, cmap="gray", vmin=0, vmax=255)
        plt.imshow(raw_image, cmap="gray")
        plt.axis('off')  # 隐藏坐标轴
        # plt.xticks(np.arange(0, raw_image.shape[1], 1))
        # plt.yticks(np.arange(0, raw_image.shape[0], 1))
        plt.show()

    return raw_image


def get_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray, n=np.ndarray([0, 0, 1]), d=1) -> np.ndarray:
    """
    计算单应矩阵,有另外一种可以不用参数n的方法
    :param K: 相机内参矩阵
    :param R: 相机的旋转
    :param t: 相机的位移
    :param n: 像平面法向量
    :param d: 像平面深度
    :return H: 返回单应矩阵
    """
    H = K @ (R + (1 / d) * np.outer(t.T, n)) @ np.linalg.inv(K)
    """
    # K = np.array([[726.28741455078, 0, 354.6496887207],
    #               [0, 726.28741455078, 186.46566772461],
    #               [0, 0, 1]])

    # R = np.array([[0.9962, -0.0872, 0],
    #               [0.0872, 0.9962, 0],
    #               [0, 0, 1]])

    # t = np.array([0, 0, 0])

    # H = get_homography(K, R, t)

    # O = np.array([354.6496887207, 186.46566772461, 1])

    # print(H @ O)
    the rotation result is:[354.64968872 186.46566772   1.        ]
    """
    return H


@singledispatch
def calcu_each_block_psf(image_blocks: np.ndarray, n: int, H: np.ndarray, overlap: int = 16,
                         vis: bool = None) -> tuple:  # diag = rad
    """
    计算每一个图像块的模糊核
    :param image_blocks:图像块数组
    :param n:分割块数(NxN)
    :param H:相机运动的单应矩阵
    :param vis:# 如果vis=False,则blocks_psf里面的每一个元素是[l,theta], # 如果vis=True,则blocks_psf里面每一个元素都是一个模糊化的图像
    :return blocks_psf: 
    """
    global kernel
    height_base, width_base = image_blocks[0, 0].shape[0] - overlap, image_blocks[0, 0].shape[1] - overlap
    blocks_psfs = np.empty((n, n), object)
    psfs_image = np.zeros((n, n), np.ndarray)

    for i in range(n):
        for j in range(n):
            # step 1: find the center coordinate of each block in Image frame
            blocki_y, blocki_x = height_base * (2 * i + 1) / 2, width_base * (2 * j + 1) / 2
            # print(blocki_x, ", ", blocki_y)
            # step 2: calculate
            l, theta = calcu_pixel_motion(H, np.array([blocki_x, blocki_y, 1]))
            if vis:
                kernel = PSF.PSFFunction(l, theta * 180 / np.pi)
                kernel.calculate_h()
                blurred = cv.filter2D(image_blocks[i, j], -1, kernel.hh, borderType=cv.BORDER_REPLICATE)
                # kernel = get_motion_psf(image_blocks[i, j].shape, theta * 180 / np.pi, l)
                # blurred = cv.filter2D(image_blocks[i, j], -1, kernel, borderType=cv.BORDER_REPLICATE)
                blocks_psfs[i, j] = blurred
            else:
                blocks_psfs[i][j] = np.array([l, theta])
                psfs_image[i][j] = kernel_compliant(kernel.hh, image_blocks[i][j].shape)

    return blocks_psfs, psfs_image


@calcu_each_block_psf.register(bool)
def _(H, image_blocks: np.ndarray, depth_image: np.ndarray, N: int,
      K: np.ndarray, Rij: np.ndarray, tij: np.ndarray, overlap: int, vis: bool = False) -> tuple:
    height_base, width_base = image_blocks[0, 0].shape[0] - overlap, image_blocks[0, 0].shape[1] - overlap
    blocks_psfs = np.zeros((N, N), object)
    psfs_image = np.zeros((N, N), np.ndarray)

    for i in range(N):
        for j in range(N):
            blocki_y, blocki_x = height_base * (2 * i + 1) / 2, width_base * (2 * j + 1) / 2
            depth = get_avedepth(depth_image, blocki_x, blocki_y, N) if not depth_image[
                int(blocki_y), int(blocki_x)] else depth_image[int(blocki_y), int(blocki_x)]
            # print("x is {0}, y is {1}, depth is {2}".format(blocki_x, blocki_y, depth))
            # 深度如果是0则取块的平均深度反之取点深度
            l, theta = calcu_pixel_motion(False, K, Rij, tij, np.array([blocki_x, blocki_y, 1]), depth * 1e-3)
            # 32 bits images should be in meters, and 16 bits should be in mm
            kernel = PSF.PSFFunction(l, np.degrees(theta))
            kernel.calculate_h()
            if vis:
                # blurred = cv.filter2D(image_blocks[i, j], -1, kernel.hh, borderType=cv.BORDER_REPLICATE)
                blurred = warp_img(image_blocks[i, j], kernel.hh)
                blocks_psfs[i, j] = blurred
            else:
                blocks_psfs[i][j] = kernel.hh
                psfs_image[i][j] = kernel_compliant(kernel.hh, image_blocks[i][j].shape)

    return blocks_psfs, psfs_image


def get_avedepth(depth: np.ndarray, x: np.int32, y: np.int32, N: np.int32) -> ndarray:
    """
    :param depth: 深度图
    :param x: 横坐标
    :param y: 纵坐标
    :param N: 多少等分
    :return ave_depth: 返回块的平均深度
    """
    extend_x = depth.shape[1] // N
    extend_y = depth.shape[0] // N
    depth_slice = depth[int(y - 1 / 2 * extend_y): int(y + 1 / 2 * extend_y),
                  int(x - 1 / 2 * extend_x): int(x + 1 / 2 * extend_x)]
    ave_depth = np.mean(depth_slice)
    return ave_depth


@singledispatch
def calcu_pixel_motion(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    计算像素运动
    :param H: Homograph
    :param point: pixel coordinate point
    :return: l(像素点移动的大小), theta(像素点移动的方向角度)
    """
    pixel_coor_bef = point
    pixel_coor_after = H @ pixel_coor_bef
    l = np.sqrt(
        (pixel_coor_after[0] - pixel_coor_bef[0]) ** 2 + (pixel_coor_after[1] - pixel_coor_bef[1]) ** 2)
    l = np.ceil(l)
    theta = math.atan2(pixel_coor_bef[1] - pixel_coor_after[1], pixel_coor_after[0] - pixel_coor_bef[0])

    return np.array([l, theta])


@calcu_pixel_motion.register(bool)
def _(H, K: np.ndarray, R: np.ndarray, t: np.ndarray, point: np.ndarray, d1: float) -> np.ndarray:
    """
    计算像素运动
    :param H: 花瓶
    :param K: 内参矩阵
    :param R: 帧间旋转矩阵
    :param t: 帧间位移
    :param point: pixel coordinate point
    :param point: 相机深度
    """
    pixel_coor_bef = point
    pixel_coor_after = K @ R @ np.linalg.inv(K) @ pixel_coor_bef + 1 / d1 * t

    l = np.sqrt(
        (pixel_coor_after[0] - pixel_coor_bef[0]) ** 2 + (pixel_coor_after[1] - pixel_coor_bef[1]) ** 2)
    l = np.ceil(l)
    theta = math.atan2(pixel_coor_bef[1] - pixel_coor_after[1], pixel_coor_after[0] - pixel_coor_bef[0])

    return np.array([l, theta])


def cls_filter(img, kernel, laplacian, gamma) -> np.ndarray:
    """
    有约束的最小二乘滤波器实现
    :param img:模糊图像
    :param kernel:模糊核
    :param laplacian:拉普拉斯算子
    :param gamma:系数什么数值效果好取哪一个
    :return:返回滤波后的图像
    """
    img_fft = np.fft.fft2(img)
    PSF_fft = np.fft.fft2(kernel, s=img.shape)
    laplacian_fft = np.fft.fft2(laplacian, s=img.shape)
    Filter = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + gamma * (np.abs(laplacian_fft) ** 2))
    deblur_fft = np.fft.fftshift(Filter * img_fft)
    deblur_fft = np.fft.ifft2(deblur_fft)
    deblur_img = np.abs(deblur_fft)
    return deblur_img


def winerFilter(img, kernel, eps, K=0.01) -> np.ndarray:
    """
    维纳滤波去运动模糊, 对kernel的形状有要求, 后面需要调整
    :param img: 灰度图像
    :param kernel: 估计出的运动模糊核
    :param eps: ???
    :param K: 噪信比
    :return: 返回处理后的图像
    """
    fftImg = np.fft.fft2(img)
    fftPSF = np.fft.fft2(kernel, s=img.shape) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF) ** 2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter


def inverseFilter(img, kernel) -> np.ndarray:
    """
    逆滤波去运动模糊, 对kernel的形状有要求, 后面需要调整
    :param img: 逆滤波的图像
    :param kernel: 估计出的运动模糊核
    :return: 返回处理后的图像
    """
    return winerFilter(img, kernel, 0)


def kernel_compliant(kernel: np.ndarray, block_shape=(30, 40)) -> np.ndarray:
    pre_height, pre_width = kernel.shape
    new_height, new_width = block_shape
    new_kernel = np.zeros(block_shape, dtype=np.float32)
    new_kernel[new_height // 2 - math.ceil(pre_height / 2): new_height // 2 + math.floor(pre_height / 2),
    new_width // 2 - math.ceil(pre_width / 2): new_width // 2 + math.floor(pre_width / 2)] = kernel
    return new_kernel


def warp_img(img, A):
    if img.ndim != 2:
        height, width, _ = img.shape
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]

        red_out = (A @ red.flatten()).reshape(height, width)
        green_out = (A @ green.flatten()).reshape(height, width)
        blue_out = (A @ blue.flatten()).reshape(height, width)

        warped = np.dstack((red_out, green_out, blue_out))
    else:
        height, width, _ = img.shape
        warped = (A @ img.flatten()).reshape(height, width)

    return warped
