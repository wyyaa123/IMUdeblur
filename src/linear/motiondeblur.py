#! /usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
# import rospy
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
import time


# from calcu_imu.msg import ImageMotion

def getMotionDsf(shape, angle, dist):
    xCenter = shape[0] // 2
    yCenter = shape[1] // 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros(shape)  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    return PSF / PSF.sum()  # 归一化


def ClsFilter(img, kernel, laplacian, gamma):
    img_fft = np.fft.fft2(img)
    PSF_fft = np.fft.fft2(kernel)
    laplacian_fft = np.fft.fft2(laplacian, s=img.shape)
    Filter = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + gamma * (np.abs(laplacian_fft) ** 2))
    deblur_fft = np.fft.ifft2(Filter * img_fft)
    deblur_img = np.abs(np.fft.fftshift(deblur_fft))
    return deblur_img


def winerFilter(input, PSF, eps, K=0.01):
    fftImg = np.fft.fft2(input)
    fftPSF = np.fft.fft2(PSF) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF) ** 2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter


def swap_diag_blocks(mat):
    h, w = mat.shape
    h1 = math.ceil(h / 2)
    w1 = math.ceil(w / 2)
    mat1 = mat[0:h1, 0:w1]
    mat2 = mat[0:h1, w1:w]
    mat3 = mat[h1:h, 0:w1]
    mat4 = mat[h1:h, w1:w]

    temp = np.copy(mat1)
    mat1 = np.copy(mat4)
    mat4 = np.copy(temp)
    temp = np.copy(mat2)
    mat2 = np.copy(mat3)
    mat3 = np.copy(temp)

    merged_mat = np.vstack((np.hstack((mat1, mat2)), np.hstack((mat3, mat4))))
    return merged_mat


def inverseFilter(img, kernel):
    return winerFilter(img, kernel, 0)


# def ImageMotionCallBack(imageMotion = ImageMotion()):
#     bridge = CvBridge()
#     cv_image = bridge.imgmsg_to_cv2(imageMotion.Image, "passthrough")
#     gray_img = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
#     PSF = getMotionDsf(gray_img.shape, imageMotion.angle, imageMotion.length)
#     laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#     deblur_img = ClsFilter(gray_img, PSF, laplacian, 0.01)
#     cv.imshow("deblur", deblur_img / 256)
#     cv.waitKey(10)

def show_imgSpectrum(img):
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dftshift = np.fft.fftshift(dft)
    result = 20 * np.log(cv.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.axis('off')
    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('original'), plt.axis('off')
    plt.show()


if __name__ == "__main__":
    gray_img = cv.imread("../../images/blur l 33.0 theta 0.png", cv.IMREAD_GRAYSCALE)
    # roi = gray_img[350:720, 875:1280]
    # show_imgSpectrum(gray_img)
    PSF = getMotionDsf(gray_img.shape, 0, 33)
    # blur_image = cv.filter2D(gray_img, -1, PSF)

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # start_time = time.time()
    # deblur_img = winerFilter(gray_img, PSF, 0.001)

    deblur_img = ClsFilter(gray_img, PSF, laplacian, 0.001)
    # swap_diag_blocks(deblur_img)
    # end_time = time.time()

    # 计算代码执行时间
    # elapsed_time = end_time - start_time

    # 打印执行时间
    # print(f"代码执行时间：{elapsed_time:.4f}秒")
    cv.imshow("deblur", deblur_img / 256)
    cv.imwrite("deblur.jpg", deblur_img)
    cv.waitKey(0)

    # rospy.init_node("deblur")

    # sub = rospy.Subscriber("/blur_img", ImageMotion, ImageMotionCallBack, queue_size=100)

    # rospy.spin()

    # rospy.loginfo("the Node has been shutdown.")
