"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023-05-29 10:13
# @Author : orCate

"""
import numpy as np

Rotation = np.array([[1, 0, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转-5°
                     [0, 2, 0],
                     [0, 0, 3]])

point = np.array([1, 1, 1])

print(Rotation @ point)
