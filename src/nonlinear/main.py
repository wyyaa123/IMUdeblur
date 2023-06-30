"""
# -*- coding: utf-8 -*-
# @Time : 2023-05-23 14:21
# @Author : orCate
"""
import numpy as np
import cv2 as cv
import Image
import time


def deblur_byIMU(H: np.ndarray, raw_image: np.ndarray, N: int = 16, overlap: int = 16) -> np.ndarray:
    """
    : param Rij: 两帧之间的旋转矩阵
    : param tij: 两帧之间的位移
    : param raw_image: 原模糊图像
    : param depth_image: 与模糊图像匹配的深度图
    : param K: 相机内参
    : param N: 原图要分割的块数
    : param overlap: 每一块的每一边要向外拓展的像素数
    """
    image_blocks = Image.segment_nimage(raw_image, N, overlap, False)

    block_psfs = Image.calcu_each_block_psf(image_blocks, N, H, False)

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for row in range(image_blocks.shape[0]):
        for col in range(image_blocks.shape[1]):
            # image_blocks[row, col] = _Image.winerFilter(image_blocks[row, col], block_psfs[row, col], 0.1, 0.01)
            image_blocks[row, col] = Image.cls_filter(image_blocks[row, col], block_psfs[row, col], laplacian, 0.001)

    deblur_image = Image.nimage_block_merge(image_blocks, N, overlap, True)
    return deblur_image


if __name__ == '__main__':
    """
    # width = 320
    # height = 160

    # x_coors = np.linspace(0, width - 1, width) # 此处注释的代码的运算结果是一个矩阵->维度{3 x (图像高 . 图像宽)}, 
    矩阵是由图像像素点坐标[X, Y, 1]组成,所以可以直接用单应矩阵(3 x 3)乘以他得到像素点变化后的像素坐标
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

    image = cv.imread("../../images/image3.png", cv.IMREAD_UNCHANGED)  # 读图, opencv库不指定图像读灰度图的话就算原图是灰度
    # 图,图像的属性也是三通道的
    # print(image.shape)
    N, extend = 16, 16

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

    # 相机内参阵主点为(619.0527171517301, 346.8334627145192)
    # Intrinsics = np.array([[815.5400764083713, 0, 619.0527171517301],
    #                        [0, 813.4639884474725, 346.8334627145192],
    #                        [0, 0, 1]])

    # 相机内参阵主点为(619.0527171517301, 346.8334627145192)
    Intrinsics = np.array([[815.5400764083713, 0, 619.0527171517301],
                           [0, 813.4639884474725, 346.8334627145192],
                           [0, 0, 1]])

    Rotation = np.array([[0.9986, -0.05234, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转3°
                         [0.05234, 0.9986, 0],
                         [0, 0, 1]])

    # Rotation = np.array([[0.9961, 0.08716, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转-5°
    #                      [-0.08716, 0.9961, 0],
    #                      [0, 0, 1]])

    # Rotation = np.array([[1, 0, 0],  # 相机旋转, 旋转矩阵 绕x轴旋转5°
    #                      [0, 0.9961, 0.08716],
    #                      [0, -0.08716, 0.9961]])

    # Rotation = np.array([[0.9961, 0, -0.08716],  # 相机旋转, 旋转矩阵 绕y轴旋转5°
    #                      [0, 1, 0],
    #                      [0.08716, 0, 0.9961]])

    # Rotation = np.array([[1, 0, 0],  # 相机旋转, 旋转矩阵
    #                      [0, 1, 0],
    #                      [0, 0, 1]])

    transion = np.array([0.03, 0, 0])  # 相机位移
    # transion = np.array([0, 0.05, 0])  # 相机位移

    beg_time = time.time()
    Homograph = Image.get_homography(Intrinsics, Rotation, transion)  # 计算单应矩阵

    # l, theta = calcu_pixel_motion(Homograph, [0, 0, 1]) # 计算像素点的变化对应的像素位移和角度(u0, v0)->(u1, v1),
    #                                                     # l是像素位移大小, theta是角度偏移
    #
    # print("l is {0}, theta is {1}".format(l, theta))
    #
    # kernel = PSF.PSFFunction(l, theta * 180 / np.pi)

    # kernel = get_motion_psf(image.shape, 180 * theta / np.pi, l) # 由l和theta计算线性模糊核
    # kernel.calculate_h()
    # blur_image = cv.filter2D(image, -1, kernel.hh) # 图像模糊
    # cv.imshow("blur", blur_image) #显示图像
    # cv.waitKey()

    # # # 分割图像成四个等大小的块 #该注释代码已弃用
    # # # block1 = image[:block_height, :block_width]
    # # # block2 = image[:block_height, block_width:]
    # # # block3 = image[block_height:, :block_width]
    # # # block4 = image[block_height:, block_width:]

    # beg_time = time.time()
    image_blocks = Image.segment_nimage(image, N, extend, False)  # 按行顺序存放,vis=True则显示分块后的图像
    # print("expired {0:.12f} seconds".format(time.time() - beg_time))
    blocks_psf = Image.calcu_each_block_psf(image_blocks=image_blocks, n=N, H=Homograph, vis=True)  # 计算每一块的模糊核,
    # 存放到blocks_psf(np.array)
    # 如果vis=False,
    # 则blocks_psf里面的每一个元素是[l,theta] 如果vis=True,则blocks_psf里面每一个元素都是一个模糊化的图像 for i in range(N): for j in range(N):
    # flag : bool = cv.imwrite("./block{0}{1}.png".format(i, j), blocks_image[i, j]) print(flag) # print(blocks_psf)
    # Raw_image = nimage_block_merge(image_blocks=image_blocks, n=N, overlap=extend, vis=False)  # 显示没有模糊化的原图拼接结果
    blur_image = Image.nimage_block_merge(blocks_psf, N, overlap=extend, vis=True) # 显示模糊化之后的拼接结果
    deblur_byIMU(Homograph, blur_image)
    # print(blur_image.shape)
    # 注意calcu_each_block_psf中的vis=True时才可用
    print("elapsed {0:.5f} seconds".format(time.time() - beg_time))
