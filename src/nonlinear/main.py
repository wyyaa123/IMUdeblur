"""
# -*- coding: utf-8 -*-
# @Time : 2023-05-23 14:21
# @Author : orCate
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import math
import os
import PSF


def segment_nimage(gray: cv.Mat, n: int, overlop: int, vis: bool = None) -> np.array:
    """
    分割图像,分成NxN份
    :param gray:灰度图
    :param n: NxN份
    :param overlop: 重叠的像素大小
    :param vis: 是否显示分割后的图像
    :return: 返回分割的图像块数组
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
            up_pixel = overlop if i else 0
            down_pixel = overlop if i + 1 and i + 1 != n else 0
            left_pixel = overlop if j else 0
            right_pixel = overlop if j + 1 and j + 1 != n else 0
            image_blocks[i, j] = gray[i * height_base - up_pixel: (i + 1) * height_base + down_pixel,
                                 j * width_base - left_pixel: (j + 1) * width_base + right_pixel]
            print("image_blocks[{0}][{1}] is ({2},{3})".
                  format(i, j, image_blocks[i, j].shape[0], image_blocks[i, j].shape[1]))

    if vis:
        # 创建一个包含多个子图的图像界面
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


def calWeight(d, k):
    """
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    """

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, block_shape, cnt: int, left_right=None):
    """
    :param img1:
    :param img2:
    :param overlap:
    :param block_shape:
    :param cnt:
    :param left_right:
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


def nimage_block_merge(image_blocks: np.array, n: int, overlap: int, vis: bool = None) -> np.array:
    """
    将(NxN)个图像块合并
    :param image_blocks:图像块数组
    :param n: 有NxN块
    :param overlap: 重叠区域
    :param vis: 是否显示
    :return: 返回合并后的图像
    """
    overlap *= 2
    # height_base, width_base = gray_block.shape[0], gray_block.shape[1]
    # raw_image = np.zeros((height_base * n, width_base * n))

    temp_imgblock = np.empty(N, object)

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
            if j == n:
                temp_img = temp_img[:, :temp_img.shape[1] - overlap // 2]
            # print("temp_img.shape is", temp_img.shape)
            # os.system("cls")
            # cv.imshow("temp_img", temp_img / 256)
            # cv.waitKey()
            temp_imgblock[i] = temp_img
            cnt += 1

    cnt = 1
    temp_img = temp_imgblock[0]
    for i in range(1, n):
        print(i)
        temp_img = imgFusion(temp_img, temp_imgblock[i], overlap, temp_imgblock[0].shape, cnt, False)
        cnt += 1

    raw_image = temp_img

    if vis:
        # plt.imshow(raw_image, cmap="gray", vmin=0, vmax=255)
        plt.imshow(raw_image, cmap="gray")
        plt.axis('on')  # 隐藏坐标轴
        # plt.xticks(np.arange(0, raw_image.shape[1], 1))
        # plt.yticks(np.arange(0, raw_image.shape[0], 1))
        plt.show()

    return raw_image


def image_fft(gray: cv.Mat, vis: bool = None) -> None:
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


def get_homography(K: np.array, R: np.array, t: np.array, n=np.array([0, 0, 1]), d=1) -> np.array:
    """
    计算单应矩阵,有另外一种可以不用参数n的方法
    :param K: 相机内参矩阵
    :param R: 相机的旋转
    :param t: 相机的位移
    :param n: 像平面法向量
    :param d: 像平面深度
    :return: 返回单应矩阵
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


def calcu_each_block_psf(image_blocks: np.array, n: int, H: np.array, vis: bool = None) -> np.array:  # diag = rad
    """
    计算每一个图像块的模糊核
    :param image_blocks:图像块数组
    :param n:分割块数(NxN)
    :param H:相机运动的单应矩阵
    :param vis:# 如果vis=False,则blocks_psf里面的每一个元素是[l,theta], # 如果vis=True,则blocks_psf里面每一个元素都是一个模糊化的图像
    :return:
    """
    height_base, width_base = image_blocks[0, 0].shape[1], image_blocks[0, 0].shape[0]
    blocks_psfs = np.empty((n, n), object)

    for i in range(n):
        for j in range(n):
            # step 1: find the center coordinate of each block in Image coordinate
            blocki_x, blocki_y = height_base * (2 * i + 1) / 2, width_base * (2 * j + 1) / 2
            # step 2: calculate 
            l, theta = calcu_pixel_motion(H, [blocki_x, blocki_y, 1])
            if vis:
                kernel = PSF.PSFFunction(l, theta * 180 / np.pi)
                kernel.calculate_h()
                blurred = cv.filter2D(image_blocks[i, j], -1, kernel.hh, borderType=cv.BORDER_REPLICATE)
                # kernel = get_motion_psf(image_blocks[i, j].shape, theta * 180 / np.pi, l)
                # blurred = cv.filter2D(image_blocks[i, j], -1, kernel, borderType=cv.BORDER_REPLICATE)
                blocks_psfs[i, j] = blurred
            else:
                blocks_psfs[i][j] = np.array([l, theta])

    return blocks_psfs


def calcu_pixel_motion(H: np.array, point: np.array) -> np.array:
    """
    计算像素运动
    :param H: Homograph
    :param point: pixel coordinate point
    """
    pixel_coor_bef = point
    pixel_coor_after = H @ pixel_coor_bef
    l = np.sqrt(
        (pixel_coor_after[0] - pixel_coor_bef[0]) ** 2 + (pixel_coor_after[1] - pixel_coor_after[1]) ** 2)
    l = np.ceil(l)
    theta = math.atan2(pixel_coor_bef[1] - pixel_coor_after[1], pixel_coor_after[0] - pixel_coor_bef[0])

    return np.array([l, theta])


def get_motion_psf(shape, angle, dist):
    """
    计算运动模糊核,该函数在N=16时会报错,后面如果可以调用matlab实现的话更好
    :param shape:图像形状
    :param angle:像素点的运动角度
    :param dist:像素点的运动距离
    :return:返回模糊核
    """
    dist = int(dist)
    xCenter = shape[0] // 2
    yCenter = shape[1] // 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros(shape)  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    # return PSF / np.linalg.norm(PSF)  # 归一化
    # return PSF / np.linalg.det(PSF)
    return PSF / dist

    # M = cv.getRotationMatrix2D((dist / 2, dist / 2), angle, 1) #已弃用
    # motion_blur_kernel = np.diag(np.ones(dist))
    # motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (dist, dist))

    # motion_blur_kernel = motion_blur_kernel / dist
    # return motion_blur_kernel


def cls_filter(img, kernel, laplacian, gamma):
    """
    有约束的最小二乘滤波器实现,后面如果可以调用matlab实现的话更好
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
    deblur_fft = np.fft.ifft2(Filter * img_fft)
    deblur_img = np.abs(np.fft.fftshift(deblur_fft))
    return deblur_img


if __name__ == '__main__':
    """
    # width = 320
    # height = 160

    # x_coors = np.linspace(0, width - 1, width) # 此处注释的代码的运算结果是一个矩阵->维度{3 x (图像高 . 图像宽)}, 矩阵是由图像像素点坐标[X, Y, 1]组成,所以可以直接用单应矩阵(3 x 3)乘以他得到像素点变化后的像素坐标
    # print("x_coors is :", x_coors)
    # y_coors = np.linspace(0, height - 1, height)
    # print("y_coors is :", y_coors)
    # indices = np.meshgrid(x_coors, y_coors)
    # print("indices is :", indices)
    # indices = np.stack(indices)
    # print("indices is :", indices)
    # indices = indices.T
    # indices = indices.reshape((-1, 2))
    # print("indices is :", indices)
    # indices = indices.T
    # indices = np.vstack((indices, np.ones(height * width)))
    # print("indices is :", indices)
    # indices /= indices[2, :]
    # indices = indices.astype('int')
    # print("indices is :", indices)

    K = np.array([[726.28741455078, 0, 354.6496887207],# 此处注释是用来计算不同的四个像素点在相同单应矩阵时的像素变化
                  [0, 726.28741455078, 186.46566772461],
                  [0, 0, 1]])
    
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    t = np.array([0.1, 0.1, 0]) 
    N = np.array([0, 0, 1])
    d = 10

    H = K @ (R + (1 / d) * np.outer(t.T, N)) @ np.linalg.inv(K)

    # print("H is", H)
    #
    # leftup = np.array([0, 0, 1])
    # leftdown = np.array([0, 159, 1])
    # rightup = np.array([319, 0, 1])
    # rightdown = np.array([319, 159, 1])
    #
    # result1 = H @ leftup
    # result2 = H @ leftdown
    # result3 = H @ rightup
    # result4 = H @ rightdown
    # print("result1 is :", result1)
    # print("result2 is :", result2)
    # print("result3 is :", result3)
    # print("result4 is :", result4)
    """

    image = cv.imread("./images/image3.png", cv.IMREAD_UNCHANGED)  # 读图, opencv库不指定图像读灰度图的话就算原图是灰度图,图像的属性也是三通道的

    N, extend = 4, 16

    # kernel = getMotionDsf(75, 60) # 测试线性去模糊的代码注释,用的是最小二乘滤波
    #                               # 对应图片blur 60 75.png意思是点移动60像素,方向是75度
    #                               # 度数以图像坐标系的x轴开始,逆时针为正,顺时针为负
    # print (kernel.shape[0])
    # print (kernel.shape[1])
    # laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # deblur_img = ClsFilter(image, kernel, laplacian, 0.001)
    # cv.imshow("deblur", deblur_img)
    # cv.waitKey()

    # image_fft(gray=image, vis=False) # 如果vis为True显示图像的频谱图

    # N: int = 2 # 图像坐标系,图像的X或者Y方向要分几块,如果N=2(2x2),则图像会被分成4等份,N=4,则图像被分为16(4x4)份

    # Intrinsics = np.array([[815.5400764083713, 0, 619.0527171517301],  # 相机内参阵主点为(619.0527171517301, 346.8334627145192)
    #                        [0, 813.4639884474725, 346.8334627145192],
    #                        [0, 0, 1]])

    Intrinsics = np.array([[815.5400764083713, 0, 640],  # 相机内参阵主点为(619.0527171517301, 346.8334627145192)
                           [0, 813.4639884474725, 360],
                           [0, 0, 1]])

    Rotation = np.array([[0.9848, -0.1736, 0],  # 相机旋转, 旋转矩阵
                         [0.1736, 0.9848, 0],
                         [0, 0, 1]])

    # Rotation = np.array([[1, 0, 0],  # 相机旋转, 旋转矩阵
    #                      [0, 1, 0],
    #                      [0, 0, 1]])

    transion = np.array([0, 0, 0])  # 相机位移

    Homograph = get_homography(Intrinsics, Rotation, transion)  # 计算单应矩阵

    leftup = np.array([0, 0, 1])
    leftdown = np.array([0, 720, 1])
    rightup = np.array([1280, 0, 1])
    rightdown = np.array([1280, 720, 1])

    result1 = Homograph @ leftup
    result2 = Homograph @ leftdown
    result3 = Homograph @ rightup
    result4 = Homograph @ rightdown
    print("result1 is :", result1)
    print("result2 is :", result2)
    print("result3 is :", result3)
    print("result4 is :", result4)

    print("the l and theta of leftup is", calcu_pixel_motion(Homograph, leftup))
    print("the l and theta of leftdown is", calcu_pixel_motion(Homograph, leftdown))
    print("the l and theta of rightup is", calcu_pixel_motion(Homograph, rightup))
    print("the l and theta of rightdown is", calcu_pixel_motion(Homograph, rightdown))

    # l, theta = calcu_pixel_motion(Homograph, [0, 0, 1]) # 计算像素点的变化对应的像素位移和角度(u0, v0)->(u1, v1), 
    #                                                     # l是像素位移大小, theta是角度偏移

    # kernel = get_motion_psf(image.shape, 180 * theta / np.pi, l) # 由l和theta计算线性模糊核

    # blur_image = cv.filter2D(image, -1, kernel) # 图像模糊
    # cv.imshow("blur", blur_image) #显示图像
    # cv.waitKey()

    # # # 分割图像成四个等大小的块 #该注释代码已弃用
    # # # block1 = image[:block_height, :block_width]
    # # # block2 = image[:block_height, block_width:]
    # # # block3 = image[block_height:, :block_width]
    # # # block4 = image[block_height:, block_width:]

    # beg_time = time.time()
    # image_blocks = segment_nimage(image, N, extend, True)  # 按行顺序存放,vis=True则显示分块后的图像
    # print("expired {0:.12f} seconds".format(time.time() - beg_time))
    # blocks_psf = calcu_each_block_psf(image_blocks=image_blocks, n=N, H=Homograph, vis=True)  # 计算每一块的模糊核,
    # 存放到blocks_psf(np.array)
    # 如果vis=False,
    # 则blocks_psf里面的每一个元素是[l,theta] 如果vis=True,则blocks_psf里面每一个元素都是一个模糊化的图像 for i in range(N): for j in range(N):
    # flag : bool = cv.imwrite("./block{0}{1}.png".format(i, j), blocks_image[i, j]) print(flag) # print(blocks_psf)
    # Raw_image = nimage_block_merge(image_blocks=image_blocks, n=N, overlap=extend, vis=True)  # 显示没有模糊化的原图拼接结果
    # blur_image = nimage_block_merge(blocks_psf, N, overlap=extend, vis=True) # 显示模糊化之后的拼接结果,
    # 注意calcu_each_block_psf中的vis=True时才可用
