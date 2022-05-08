from turtle import color
import cv2
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

infra1_1262 = cv2.imread('./Homework/HW1-1-data/infra1_1262.jpg')
infra1_1755 = cv2.imread('./Homework/HW1-1-data/infra1_1755.jpg')
infra1_1131 = cv2.imread('./Homework/HW1-1-data/infra1_1131.jpg')
infra1_0000 = cv2.imread('./Homework/HW1-1-data/infra1_0000.jpg')

infra2_1262 = cv2.imread('./Homework/HW1-1-data/infra1_1262.jpg')
infra2_1755 = cv2.imread('./Homework/HW1-1-data/infra1_1755.jpg')
infra2_1131 = cv2.imread('./Homework/HW1-1-data/infra1_1131.jpg')
infra2_0000 = cv2.imread('./Homework/HW1-1-data/infra1_0000.jpg')

