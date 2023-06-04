# -*- coding: utf-8 -*-
import sys
import time
sys.path.append("/home/wyyaa123/IMUdeblur/")
import rospy
import lib.Image as _Image
import lib.PSF as _PSF
import lib.frame as _Frame
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
# import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

def quaternion_to_Rotation(qu: np.ndarray) -> np.ndarray:
    R = np.zeros((3, 3), dtype=float)
    R[0, 0] = qu[0] ** 2 + qu[1] ** 2 - qu[2] ** 2 - qu[3] ** 2
    R[0, 1] = 2 * (qu[1] * qu[2] - qu[0] * qu[3])
    R[0, 2] = 2 * (qu[1] * qu[3] + qu[0] * qu[2])
    R[1, 0] = 2 * (qu[1] * qu[2] + qu[0] * qu[3])
    R[1, 1] = qu[0]**2 - qu[1] ** 2 + qu[2] ** 2 - qu[3] ** 2
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

def deblur_byIMU(Rij: np.ndarray, tij: np.ndarray, raw_image: cv.Mat, depth_image: cv.Mat, K: np.ndarray, N: int = 16, overlap: int = 16) -> cv.Mat:
    image_blocks = _Image.segment_nimage(raw_image, N, overlap, False)
    block_psfs = _Image.calcu_each_block_psf(False, image_blocks, depth_image, N, K, Rij, tij, overlap)
    
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for row in image_blocks.shape[0]:
        for col in image_blocks.shape[1]:
            image_blocks[row, col] = _Image.cls_filter(image_blocks[row, col], block_psfs[row, col], laplacian, 0.001)
    
    deblur_image = _Image.nimage_block_merge(image_blocks, N, overlap, False)
    return deblur_image

def img0_callback(img0: Image):
    global t_i, R_i, depths, frames, img0_pub, depth_image
    
    frame_flag: bool = False
    depth_flag: bool =  False
    
    R_j = np.eye(3, dtype=float)
    t_j = [0, 0, 0]
    
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img0, "mono8")
    
    # print("img0的时间戳为", img0.header.stamp.to_sec())
    
    for frame in frames: #匹配帧
        # print("frame的时间戳为", frame.timestamp)
        if np.abs(img0.header.stamp.to_sec() - frame.timestamp) < 0.03:
            # print("对应帧匹配成功!")
            frame_flag = True
            R_j = frame.Rj
            t_j = frame.tj
            
            print(frame.timestamp)
            np_frame = np.array(frames)      
            temp_frame = np_frame == frame
            # print(temp_depth)
            indices = np.where(temp_frame == True)
            # print(indices)
            frames[:int(indices[0])] = []
            break
    
    for depth in depths: #匹配深度图
        # print("深度图的时间戳为", depth.header.stamp.to_sec())
        if np.abs(img0.header.stamp.to_sec() - depth.header.stamp.to_sec()) < 0.03: # 这里如果没有找到匹配的深度图会使用上一帧的深度图
            # print("对应深度图匹配成功!")
            
            # 这一部分这么写是因为只有np数组判断后的结果仍然是一个布尔类型的数组, 列表里哪个元素符合条件, 然后把他之前的所有数据清空
            # print("深度图的长度为", np.size(depths))
            np_depth = np.array(depths)      
            temp_depth = np_depth == depth
            # print(temp_depth)
            indices = np.where(temp_depth == True)
            # print(indices)
            depths[:int(indices[0])] = []
            
            depth_flag = True
            depth_image = bridge.imgmsg_to_cv2(depth, "passthrough") #16UC1
            depth_image.astype(np.uint16) #转为无符号16位
            # print(depth_image.shape)
            break
        
    if not depth_flag: # 直接全部清空
        depths = []
        
    # print("the size of depths is", np.size(depths))
        
    if frame_flag: #得到深度图和匹配帧
        print("OK!")
        
        Rij  = R_j * np.linalg.inv(R_i)
        tij = t_j - Rij @ t_i
        # deblur_image = None
        # deblur_image = deblur_byIMU(Rij, tij, cv_image, depth_image, K, 16, 16)  
        # deblur_image_msg = bridge.cv2_to_imgmsg(deblur_image, "passthrough")
        # img0_pub.publish(deblur_image_msg)
        # pass
        print("Rij is", Rij)
        print("Ri is", R_i)
        print("Rj is", R_j)
        print("tij is", tij)
        print("ti is", t_i)
        print("tj is", t_j)
        depth_msg = bridge.cv2_to_imgmsg(depth_image, "passthrough")
        img0_pub.publish(depth_msg)
    else:
        print("game over!")
        # pass
    
    R_i = R_j
    t_i = t_j
   
    left_imgs.append(img0)
    if np.size(left_imgs) > 2:
        left_imgs.pop(0)
    
    # pass
    # print("img0 callback!")

def img1_callback(img1: Image):    
    global img1_pub
    
    right_imgs.append(img1)
    if np.size(right_imgs) > 2:
        right_imgs.pop(0)
        
    img1_pub.publish(img1)
    # pass
    # print("img1 callback!")

def depth_callback(depth: Image):
           
    depths.append(depth)
    if np.size(depths) > 100:
        depths.pop(0)
    # pass
    # print("depth callback!") 
    
    
def imu_callback(pose_j: Odometry):
    # global R_i, t_i
    # print("imu callback!")
    global frames
    R_j = quaternion_to_Rotation(np.array([pose_j.pose.pose.orientation.w, pose_j.pose.pose.orientation.x, 
                                           pose_j.pose.pose.orientation.y, pose_j.pose.pose.orientation.z]))
    
    # print("R_j is", R_j)
    
    t_j = np.array([pose_j.pose.pose.position.x, pose_j.pose.pose.position.y, pose_j.pose.pose.position.z])
    
    frame_j = _Frame.Frame(pose_j.header.stamp.to_sec(), Rj=R_j, tj=t_j)
    
    frames.append(frame_j)
    if np.size(frames) > 1e2:
        frames.pop(0)
    # print("t_j is", t_j)
    
    # Rij = R_j @ np.linalg.inv(R_i)
    # tij = t_j - Rij @ t_i
    
    # R_i = R_j
    # t_i = t_j
    
    # print("Rij is ", Rij)
    # print("tij is", tij)
    
if __name__ == "__main__":          
    rospy.init_node("deblur_node") #初始化ros节点  

    # pose_i = Odometry()
    R_i = np.eye(3, dtype=float)
    t_i = np.array([0, 0, 0], dtype=float)
    depth_image = np.ones((400, 640), dtype=np.uint16)
    
    frames = []
    depths = []
    left_imgs = []
    right_imgs = []
    
    K = np.array([[407.76044960599006, 0, 309.58075680219565],
                  [0, 406.5997999128962, 193.42444301176783],
                  [0, 0, 1]])
    
    img0_sub = rospy.Subscriber("/stereo_inertial_publisher/left/image_rect", Image, img0_callback)
    img1_sub = rospy.Subscriber("/stereo_inertial_publisher/right/image_rect", Image, img1_callback)
    depth_sub = rospy.Subscriber("/stereo_inertial_publisher/stereo/depth", Image, depth_callback)
    imu_propagate_sub = rospy.Subscriber("/vins_estimator/imu_propagate", Odometry, imu_callback)
    
    img0_pub = rospy.Publisher("/left_image", Image, queue_size=10)
    img1_pub = rospy.Publisher("/right_image", Image, queue_size=10)
    
    rospy.spin()
