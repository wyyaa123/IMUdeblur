import numpy as np

def get_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray, n=np.array([0, 0, 1]), d=1) -> np.ndarray:
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

def get_pose(K: np.ndarray, R: np.ndarray, t: np.ndarray, d: np.float32, prev_point: np.ndarray = np.array([1, 1, 1])):
    next_point = K @ R @ np.linalg.inv(K) @ prev_point + 1 / d * K @ t
    return next_point

if __name__ == '__main__':
    
    K = np.array([[815.5400764083713, 0, 619.0527171517301],
                  [0, 813.4639884474725, 346.8334627145192],
                  [0, 0, 1]])
    
    R = np.array([[0.9986, -0.05234, 0],  # 相机旋转, 旋转矩阵 绕z轴旋转3°
                  [0.05234, 0.9986, 0],
                  [0, 0, 1]])
    
    t = np.array([0.1, 0, 0])
    
    prev_point = np.array([1, 1, 1])
    
    H = get_homography(K, R, t, d=2)
    next_point1 = H @ prev_point
    print(next_point1)
    next_point2 = get_pose(K, R, t, 2, prev_point)
    print(next_point2)

