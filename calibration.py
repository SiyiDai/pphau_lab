import cv2
import numpy as np
import pyrealsense2 as rs


pattern_size = (6, 9)

samples = []

rosbag_path = './Homework/HW1-2-data/20220405_220626.bag'
# Create a config object
config = rs.config()
# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, rosbag_path)
# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
# Create pipeline
pipeline = rs.pipeline()
# Start streaming from file
pipeline.start(config)

while True:

    frames = pipeline.wait_for_frames()
    res, corners = cv2.findChessboardCorners(frames, pattern_size)

    frames = frames[0]
    img_show = np.copy(frames)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, 'Samples captured: %d' % len(samples), (0, 
    40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show)

    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append((cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY), 
        corners))
    elif k == 27:
        break

frames.release()
cv2.destroyAllWindows()