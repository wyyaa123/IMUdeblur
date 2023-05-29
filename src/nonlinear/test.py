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

image1 = cv.imread("../../images/blur.png", cv.IMREAD_GRAYSCALE)
image2 = cv.imread("../../images/blur l 33.0 theta 0.png", cv.IMREAD_GRAYSCALE)

Image.image_fft(image1, True)
Image.image_fft(image2, True)
