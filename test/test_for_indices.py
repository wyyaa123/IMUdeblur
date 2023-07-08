import numpy as np

if __name__ == '__main__':
    width = 320
    height = 160

    x_coors = np.linspace(0, width - 1, width) # 此处注释的代码的运算结果是一个矩阵->维度{3 x (图像高 . 图像宽)}, 
    # 矩阵是由图像像素点坐标[X, Y, 1]组成,所以可以直接用单应矩阵(3 x 3)乘以他得到像素点变化后的像素坐标
    print("x_coors is :", x_coors)
    y_coors = np.linspace(0, height - 1, height)
    print("y_coors is :", y_coors)
    indices = np.meshgrid(x_coors, y_coors)
    print("indices is :", indices)
    indices = np.stack(indices)
    print("indices is :", indices)
    indices = indices.T
    indices = indices.reshape((-1, 2))
    print("indices is :", indices)
    indices = indices.T
    indices = np.vstack((indices, np.ones(height * width)))
    print("indices is :", indices)
    indices /= indices[2, :]
    indices = indices.astype('int')
    print("indices is :", indices)
