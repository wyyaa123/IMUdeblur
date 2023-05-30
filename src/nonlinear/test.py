"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023-05-29 10:13
# @Author : orCate

"""
import Image
import PSF
import cv2 as cv
import numpy as np

"""
Intrinsics = np.array([[815.5400764083713, 0, 619.0527171517301],
                    [0, 813.4639884474725, 346.8334627145192],
                    [0, 0, 1]])

Rotation = np.array([[1, 0, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转-5°
                    [0, 1, 0],
                    [0, 0, 1]])

transion = np.array([0.04, 0, 0])  # 相机位移

point = np.array([0, 0, 1])

Homograph = Image.get_homography(Intrinsics, Rotation, transion)

l, theta = Image.calcu_pixel_motion(Homograph, point)

print("l is {0}, theta is {1}".format(l, theta))
"""

kernel = PSF.PSFFunction(33, 0)

kernel.calculate_h()

image = cv.imread("../../images/blur.png", cv.IMREAD_GRAYSCALE)

# blur_image = cv.filter2D(image, -1, kernel.hh, borderType=cv.BORDER_REPLICATE)

# cv.imwrite("./blur l {0} theta {1}°.png".format(l, theta * 180 / np.pi), blur_image)
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
clear_image = Image.cls_filter(image, kernel.hh, laplacian, 0.01)
cv.imshow("clear_image", clear_image / 256)
cv.waitKey()