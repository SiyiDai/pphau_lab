import cv2
from cv2 import StereoBM
import numpy as np

### 1.Stereo Reconstruction and laser-pattern (workload 1 student):
# In this exercise, we will have a look over the 
# 1. Read the color, infrared1, infrared2 images in the folder Homework/HW-1-data (images with numbers (1262, 1755, 1131, 0000))
# 2. Use OpenCV Stereo Block Matching to find the disparity map, then use the equation for depth to calculate the estimated depth map. You could assume that (focal_length=970 mm, baseline=50 mm) 
# 3. Use OpenCV to visualize the reconstructed depth image along with the infrared images using `cv2.imshow`.
# 4. What is the difference between the depth quality with respect to 
#      1. planes with texture (Checkerboard) vs. planes without texture (the PC case)
#      2. with laser pattern (1262,1755) vs no laser-pattern (0000,1131) 

color_1262 = cv2.imread('./Homework/HW1-1-data/color1262.jpg')
color_1755 = cv2.imread('./Homework/HW1-1-data/color1755.jpg')
color_1131 = cv2.imread('./Homework/HW1-1-data/color1131.jpg')
color_0000 = cv2.imread('./Homework/HW1-1-data/color0000.jpg')

# left
infra1_1262 = cv2.imread('./Homework/HW1-1-data/infra1_1262.jpg', 0)
infra1_1755 = cv2.imread('./Homework/HW1-1-data/infra1_1755.jpg', 0)
infra1_1131 = cv2.imread('./Homework/HW1-1-data/infra1_1131.jpg', 0)
infra1_0000 = cv2.imread('./Homework/HW1-1-data/infra1_0000.jpg', 0)

# right
infra2_1262 = cv2.imread('./Homework/HW1-1-data/infra2_1262.jpg', 0)
infra2_1755 = cv2.imread('./Homework/HW1-1-data/infra2_1755.jpg', 0)
infra2_1131 = cv2.imread('./Homework/HW1-1-data/infra2_1131.jpg', 0)
infra2_0000 = cv2.imread('./Homework/HW1-1-data/infra2_0000.jpg', 0)


img_L_1 = infra1_1131
img_R_1 = infra2_1131
#cv2.normalize(img_L, img_L, 0, 1, cv2.NORM_MINMAX)
#cv2.normalize(img_R, img_R, 0, 1, cv2.NORM_MINMAX)


focal_length = 970      # lense focal length
baseline = 50           # distance in mm between the two cameras
disparities = 96        # num of disparities to consider
block = 31              # block size to match
units = 0.001           # depth units

stereo = cv2.StereoBM_create(numDisparities=disparities,
                          blockSize=block)

# stereo = cv2.StereoBM_create()

disparity = stereo.compute(img_L_1, img_R_1)
disparity_1 = stereo.compute(infra1_0000, infra2_0000)
disparity_2 = stereo.compute(infra1_1262, infra2_1262)
disparity_3 = stereo.compute(infra1_1755, infra2_1755)

valid_pixels = disparity > 0
valid_pixels_1 = disparity_1 > 0
valid_pixels_2 = disparity_2 > 0
valid_pixels_3 = disparity_3 > 0
# calculate depth data
depth = np.zeros(shape=infra1_0000.shape).astype("float")
depth_1 = np.zeros(shape=infra1_0000.shape).astype("float")
depth_2 = np.zeros(shape=infra1_0000.shape).astype("float")
depth_3 = np.zeros(shape=infra1_0000.shape).astype("float")
#depth[valid_pixels] = (focal_length * baseline) / (disparity[valid_pixels])
depth[valid_pixels] = (focal_length * baseline * units**2) / (disparity[valid_pixels]).astype('uint8')
depth_1[valid_pixels_1] = (focal_length * baseline * units**2) / (disparity_1[valid_pixels_1]).astype('uint8')
depth_2[valid_pixels_2] = (focal_length * baseline * units**2) / (disparity_2[valid_pixels_2]).astype('uint8')
depth_3[valid_pixels_3] = (focal_length * baseline * units**2) / (disparity_3[valid_pixels_3]).astype('uint8')

#cv2.normalize(depth, depth, -255, 0, cv2.NORM_MINMAX)
#depth = depth + 255
#depth = depth.astype('uint8')
#cv2.normalize(depth_1, depth_1, -255, 0, cv2.NORM_MINMAX)
#depth_1 = depth_1 + 255
#depth_1 = depth_1.astype('uint8')
#cv2.normalize(depth_2, depth_2, -255, 0, cv2.NORM_MINMAX)
#depth_2 = depth_2 + 255
#depth_2 = depth_2.astype('uint8')
#cv2.normalize(depth_3, depth_3, -255, 0, cv2.NORM_MINMAX)
#depth_3 = depth_3 + 255
#depth_3 = depth_3.astype('uint8')

cv2.imshow('disparity', depth)
cv2.imshow('disparity_1', depth_1)
cv2.imshow('disparity_2', depth_2)
cv2.imshow('disparity_3', depth_3)

cv2.imwrite('disparity.png', depth)
cv2.imwrite('disparity_1.png', depth_1)
cv2.imwrite('disparity_2.png', depth_2)
cv2.imwrite('disparity_3.png', depth_3)


cv2.waitKey(-1)

