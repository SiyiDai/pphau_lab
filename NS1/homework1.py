import cv2
from cv2 import StereoBM
import numpy as np

# 1.Stereo Reconstruction and laser-pattern (workload 1 student):
# In this exercise, we will have a look over the
# 1. Read the color, infrared1, infrared2 images in the folder Homework/HW-1-data (images with numbers (1262, 1755, 1131, 0000))
# 2. Use OpenCV Stereo Block Matching to find the disparity map, then use the equation for depth to calculate the estimated depth map. 
# You could assume that (focal_length=970 mm, baseline=50 mm)
# 3. Use OpenCV to visualize the reconstructed depth image along with the infrared images using `cv2.imshow`.
# 4. What is the difference between the depth quality with respect to
#      1. planes with texture (Checkerboard: 1262 & 1131) vs. planes without texture (the PC case: 1755 & 0000)
#               Answer: In this case we choose 1131 (checkerboard) and 0000 (the PC case) for comparison, in order to avoid the 
#                       influence of laser pattern. The depth quality in 1131 is better than the one in 0000. 
#                       The difference mainly lay in the part of desk surface and the board. 1131, with the checkerboard, is able to 
#                       distinguish the desk surface and the board, while 0000 can not.
#      2. with laser pattern (1262,1755) vs no laser-pattern (0000,1131)
#               Answer: We choose 1131 and 1262 for comparison, in order to avoid the influence of checkerboard. The image with
#                       laser pattern can overcome the impact from color contraction. In 1131 (no laser-pattern), the carboard box 
#                       and the pc case are in the same surface relate to the camera, but they have been detected in different layer,
#                       while in 1262 (with laser pattern), these two objects are disticted successfully.  


def disparity_map(
    img_L, img_R, focal_length=970, baseline=50, disparities=96, block=31
):
    stereo = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)

    disparity = stereo.compute(img_L, img_R)
    valid_pixels = disparity > 0

    # calculate depth data
    depth = np.zeros(shape=img_L.shape).astype("float32")
    depth[valid_pixels] = (focal_length * baseline) / (disparity[valid_pixels])
    return depth


def main():
    # left
    infra1_1262 = cv2.imread(
        "./Homework/HW1-1-data/infra1_1262.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra1_1755 = cv2.imread(
        "./Homework/HW1-1-data/infra1_1755.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra1_1131 = cv2.imread(
        "./Homework/HW1-1-data/infra1_1131.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra1_0000 = cv2.imread(
        "./Homework/HW1-1-data/infra1_0000.jpg", cv2.IMREAD_GRAYSCALE
    )

    # right
    infra2_1262 = cv2.imread(
        "./Homework/HW1-1-data/infra2_1262.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra2_1755 = cv2.imread(
        "./Homework/HW1-1-data/infra2_1755.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra2_1131 = cv2.imread(
        "./Homework/HW1-1-data/infra2_1131.jpg", cv2.IMREAD_GRAYSCALE
    )
    infra2_0000 = cv2.imread(
        "./Homework/HW1-1-data/infra2_0000.jpg", cv2.IMREAD_GRAYSCALE
    )

    depth_1262 = disparity_map(infra1_1262, infra2_1262)
    depth_1755 = disparity_map(infra1_1755, infra2_1755)
    depth_1131 = disparity_map(infra1_1131, infra2_1131)
    depth_0000 = disparity_map(infra1_0000, infra2_0000)

    while True:
        cv2.imshow("Disparity 1262", depth_1262)
        cv2.imshow("Disparity 1755", depth_1755)
        cv2.imshow("Disparity 1131", depth_1131)
        cv2.imshow("Disparity 0000", depth_0000)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
