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
from platform import node
import pyrealsense2 as rs
import pyrender
import trimesh
import numpy as np
import cv2
import time
import threading

PATTERN_SIZE = (6, 9)
EDGE_LENGTH = 4


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
    return depth_color_image


def get_aligned_images(frames, color_image, colorizer):
    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)

    # Update color and depth frames:
    aligned_depth_frame = aligned_frames.get_depth_frame()
    colorized_depth = np.asanyarray(
        colorizer.colorize(aligned_depth_frame).get_data()
    )

    # Show the two frames together:
    aligned_images = np.hstack((color_image, colorized_depth))

    return aligned_images


def get_color_camera_intrinsics(profile):
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    return color_intrinsics


def get_depth_camera_intrinsics(profile):
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    return depth_intrinsics


def get_camera_matrix(intrin):
    # Get camera metrix from camera intrinsics
    camera_matrix = np.array(
        [[intrin.fx, 0, intrin.ppx], [0, intrin.fy, intrin.ppy], [0, 0, 1]]
    )
    return camera_matrix


def chessboard_detect(color_image):
    # Using opencv we want Find its corners (use cv2.findChessboardCorners,
    # and cv2.cornersSubPix). then use cv2.drawChessboardCorners to overlay
    # the detections on the colored image
    img_show = np.copy(color_image)
    res, corners = cv2.findChessboardCorners(img_show, PATTERN_SIZE, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    cv2.cornerSubPix(grayscale_image, corners, (10, 10), (-1, -1), criteria)
    cv2.drawChessboardCorners(img_show, (6, 9), corners, res)
    cv2.imshow("Chessboard Stream", img_show)

    return corners


def translation_calculation(corners, intrin):
    # From the previous step, you will have 2D/3D correspondences for the
    # corners. Use cv2.solvePnP to estimate the object to camera translation
    # and rotation vectors.
    camera_matrix = get_camera_matrix(intrin)
    dist_coefs = np.asanyarray(intrin.coeffs)
    
    # pattern_points is the 3D object points
    pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
    pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)

    ret, rvec, tvec = cv2.solvePnP(
        pattern_points,
        corners,
        camera_matrix,
        dist_coefs,
        None,
        None,
        False,
        cv2.SOLVEPNP_ITERATIVE,
    )
    
    # image_points is the 2D coordinates in a image
    img_points, _ = cv2.projectPoints(
        pattern_points, rvec, tvec, camera_matrix, dist_coefs
    )

    ## Generate homogenous transformation matrix
    rmat = cv2.Rodrigues(rvec)[0]
    # print('rmat', rmat)
    rmat = rmat * -1
    # print('rvec', rvec)
    # print('rmat', rmat)
    homo_trans = np.hstack([rmat, tvec])
    base = [0, 0, 0, 1]
    homo_trans = np.vstack([homo_trans, base])
    
    # homo_trans_T = np.vstack([rmat.T, tvec.T])
    # # print(homo_trans_T)
    # base = np.array([0, 0, 0, 1]).reshape((4, -1))
    # # print(base)
    # homo_trans_T = np.hstack([homo_trans_T, base])

    return homo_trans, img_points


def point_project(img_points, color_image):
    # Extra: Use opencv drawing utils and perspective projection function to
    # draw a 3D axis, and a cropping mask for the board. Useful functions here
    # could be cv2.line,cv2.projectPoints,cv2.fillPoly.

    # Show image with drawn polygon
    img_show = np.copy(color_image)
    img_points_mat = np.asanyarray(img_points).reshape(-1, 2)
    
    r_idx = np.where(img_points_mat[:, 1] == img_points_mat[:, 1].max())
    l_idx = np.where(img_points_mat[:, 1] == img_points_mat[:, 1].min())
    t_idx = np.where(img_points_mat[:, 0] == img_points_mat[:, 0].max())
    b_idx = np.where(img_points_mat[:, 0] == img_points_mat[:, 0].min())
    
    # for opencv 4.5.5 or higher, the type of vertex should be int
    r_vertex = tuple(img_points_mat[r_idx][0].astype(int))
    l_vertex = tuple(img_points_mat[l_idx][0].astype(int))
    t_vertex = tuple(img_points_mat[t_idx][0].astype(int))
    b_vertex = tuple(img_points_mat[b_idx][0].astype(int))
    
    # give the boundary of inner 54 blocks
    img_show = cv2.line(img_show, r_vertex, t_vertex, (255, 0, 0), 2)
    img_show = cv2.line(img_show, t_vertex, l_vertex, (255, 0, 0), 2)
    img_show = cv2.line(img_show, l_vertex, b_vertex, (255, 0, 0), 2)
    img_show = cv2.line(img_show, b_vertex, r_vertex, (255, 0, 0), 2)
    

    # Show white-filled polygon with black background
    bw_frames = np.zeros(img_show[:, :, 0].shape)
    counter = np.array(
        [
            img_points_mat[r_idx][0],
            img_points_mat[t_idx][0],
            img_points_mat[l_idx][0],
            img_points_mat[b_idx][0],
        ],
        dtype=np.int32,
    )

    bw_frames = cv2.fillPoly(bw_frames, [counter], (255, 255, 255))
    return bw_frames


# def mesh_creation(homo_trans):
#     # Generate Trimesh based on object points and homogenous tfs matrix
#     board_center = cal_board_center()
#     sm = trimesh.creation.box(extents=[36, 24, 0.01])
#     sm.visual.vertex_colors = [1, 0, 0]
#     tfs = np.tile(homo_trans, (len(board_center), 1, 1))
#     tfs[:, :3, 3] = board_center
#     mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
#     return mesh

def mesh_creation(homo_trans):
    # Generate Trimesh based on object points and homogenous tfs matrix
    board_center = cal_board_center()
    sm = trimesh.creation.box(extents=[24, 36, 0.01])
    sm.visual.vertex_colors = [1, 0, 0]
    tfs = np.tile(np.eye(4), (len(board_center), 1, 1))
    # tfs[:, :3, 3] = board_center
    mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    return mesh


def cal_board_center(pattern_size=PATTERN_SIZE, edge_length=EDGE_LENGTH):
    # calculate the board center with pattern size and edge length
    x = pattern_size[0] * edge_length / 2
    y = pattern_size[1] * edge_length / 2
    z = 0
    return [x, y, 0]


def set_scene(mesh, homo_trans, intrinsic):
    # set the scene with pyrender
    scene = pyrender.Scene()
    # cam = pyrender.IntrinsicsCamera(fx = intrinsic.fx, fy = intrinsic.fy, cx = intrinsic.ppx, cy = intrinsic.ppy)
    chess_node = pyrender.Node(mesh=mesh, matrix=homo_trans)
    # camera_node = pyrender.Node(camera=cam, matrix=homo_trans)
    scene.add_node(chess_node)
    # scene.add_node(camera_node)
    # V = pyrender.Viewer(scene, run_in_thread=True)
    return scene, [chess_node] #, camera_node]

# def render_3d(obj_points):
#     mesh = pyrender.Mesh.from_points(obj_points)
#     scene = pyrender.Scene()
#     scene.add(mesh)
#     v = pyrender.Viewer(scene, use_raymond_lighting = True, run_in_thread = True)
    


def pipeline_load(rosbag_path):
    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used
    # by the pipeline through playback.
    rs.config.enable_device_from_file(config, rosbag_path)
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Create pipeline
    pipeline = rs.pipeline()
    # Start streaming from file
    pipeline.start(config)
    profile = pipeline.get_active_profile()
    color_intrin = get_color_camera_intrinsics(profile)
    return pipeline, color_intrin


def event_loop(V, node, pose):
    i = 0
    while V.is_active:
        V.render_lock.acquire()
        for n in V.scene.mesh_nodes:
            V.scene.set_pose(n, pose)
        V.render_lock.release()
        i += 0.01
        time.sleep(0.1)


def main():
    rosbag_path = "../Homework/HW1-2-data/20220405_220626.bag"
    pipeline, color_intrin = pipeline_load(rosbag_path)

    flag = False
    # rotate = trimesh.transformations.rotation_matrix(angle=np.radians(10.0), direction=[0,1,0], point=[0,0,0])
    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        color_image, colorizer = get_color_images(frames)
        depth_color_image = get_depth_images(frames, colorizer)
        aligned_images = get_aligned_images(frames, color_image, colorizer)
        corners = chessboard_detect(color_image)
        homo_trans, img_points = translation_calculation(
            corners, intrin=color_intrin
        )
        # TODO: debug point_project
        bw_frames = point_project(img_points, color_image)

        mesh = mesh_creation(homo_trans)
        if flag == False:
            scene, node = set_scene(mesh, homo_trans, color_intrin)
            V = pyrender.Viewer(scene, run_in_thread=True, show_mesh_axes = True) #, rotate = True, rotate_axis=[0,0,1])
            flag = True
        else:
            # update the pose only when the scene is set
            # V = pyrender.Viewer(scene, run_in_thread=True, show_mesh_axes = True, rotate = False)
            V.render_lock.acquire()
            scene.set_pose(node[0], pose=homo_trans)
            # scene.set_pose(node[1], pose=homo_trans)
            V.render_lock.release()
        # TODO: align the rotation of pyrender View/camera/scene
        # TODO: compare the difference
        # Due to projection, the movement of the chessboard in scene viewer is less than that in the Black-white mask screen.
        # Render image in opencv window       
        cv2.imshow("Depth Stream", depth_color_image)
        cv2.imshow("Color Stream", color_image)
        cv2.imshow("Aligned Stream", aligned_images)
        cv2.imshow("Black-white mask", bw_frames)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__ == "__main__":
    main()