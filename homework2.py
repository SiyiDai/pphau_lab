# 2. Object Twin (workload 3 students):
# In this exercise, we will load a realsense-viewer rosbag recording, then use opencv and pyrender to create a twin of a moving checkerboard.

# Loading color and depth data:
# Use pyrealsense2 to read the bagfile and acquire color, depth, aligned depth to color, color camera intrinsics, depth camera intrinsics. (Show the images in a loop using cv2.imshow)
#
# Checkerboard detection and tracking:
# The checkerboard has a 6x9 pattern where each square has an edge length of 4 cm.
# Using opencv we want Find its corners (use cv2.findChessboardCorners, and cv2.cornersSubPix). then use cv2.drawChessboardCorners to overlay the detections on the colored image
# From the previous step, you will have 2D/3D correspondences for the corners. Use cv2.solvePnP to estimate the object to camera translation and rotation vectors.
# Extra: Use opencv drawing utils and perspective projection function to draw a 3D axis, and a cropping mask for the board. Useful functions here could be cv2.line,cv2.projectPoints,cv2.fillPoly.
#
# Modeling the checkerboard in pyrender:
# Using pyrender create a scene with camera and a Box mesh corresponding to the checkerboard.
# Notes:
# You will need to scale the box and shift its center to match the checkerboard 3d coordinate system in opencv
# To convert from opencv camera to pyrender camera in you system you may need to rotate your objects by 90 degees around the X-axis (depending on your implementation)
#
# Visualization:
# In the loop, update the mesh pose with the updated pose of the checkerboard
# Compare the rendered depth value to the actual algined_depth values we got from realsense.


# First import library
from matplotlib.pyplot import get
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

PATTERN_SIZE = (6, 9)

def get_color_images(frames):
    # Get color frame
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    # Create colorizer object
    colorizer = rs.colorizer()
    return color_image, colorizer

def get_depth_images(frames, colorizer):
    # Get depth frame
    depth_frame = frames.get_depth_frame()
    # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)
    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return depth_color_image, depth_image

def get_aligned_images(frames, color_image, colorizer):
    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)

    # Update color and depth frames:
    aligned_depth_frame = aligned_frames.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # Show the two frames together:
    aligned_images = np.hstack((color_image, colorized_depth))

    return aligned_images

def chessboard_detect(color_image):
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    found, corners = cv2.findChessboardCorners(grayscale_image, PATTERN_SIZE, None)
    if found:
        cv2.cornerSubPix(grayscale_image, corners, (11, 11), (-1, -1), stop_criteria)
        cv2.drawChessboardCorners(color_image, (9, 6), corners, found)
        cv2.imshow("Chessboard Stream", color_image)

def main():
    rosbag_path = './Homework/HW1-2-data/20220405_220626.bag'
    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, rosbag_path)
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Create pipeline
    pipeline = rs.pipeline()
    # Start streaming from file
    pipeline.start(config)
    # # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        color_image, colorizer = get_color_images(frames)
        depth_color_image, depth_image = get_depth_images(frames, colorizer)
        aligned_images = get_aligned_images(frames, color_image, colorizer)
        chessboard_detect(color_image)

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        cv2.imshow("Color Stream", color_image)
        cv2.imshow("Aligned Stream", aligned_images)
        

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()