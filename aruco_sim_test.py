from simulator import Camera, RigidObject
from utils import *
import numpy as np
import quaternion
import cv2
from cv2 import aruco
import pybullet as p
import pybullet_data
import argparse
import json
import os


with open('config/config.json', 'r') as f:
    cfg = json.load(f)
p.connect(p.GUI)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF('plane.urdf')

board_name = 'ArUco_DICT_7X7_50_5_7_50_10_0'
# board_name = 'ChArUco_DICT_7X7_50_5_7_50_10'
dict_str = '_'.join(board_name.split('_')[1:4])
col, row, length, sep, start_id = map(lambda x: int(x), board_name.split('_')[4:])
# col, row, length, sep = map(lambda x: int(x), board_name.split('_')[4:])
aruco_dict = aruco.Dictionary_get(ARUCO_DICT[dict_str])
board = aruco.GridBoard_create(col, row, length/1000, sep/1000, aruco_dict, start_id)
# board = aruco.CharucoBoard_create(col, row, (length+sep*2)/100, length/100, aruco_dict)
board_params = aruco.DetectorParameters_create()

vis_params, col_params, body_params = get_param_template()
col_params['fileName'] = vis_params['fileName'] = os.path.join('assets', board_name+'.obj')
board_obj = RigidObject(board_name, vis_params, col_params, body_params)
board_obj.set_pose_from_pos_and_quat([0, 0, 0.1], [0, 0, 0, 1])
ground_truth_base = np.array([-(2.5*length+2*sep)/1000, -(3.5*length+3*sep)/1000, 0.1])
cam = Camera(**cfg['camera'])
# mtx = cam.intrinsic  # cam.intrinsic [-0.14504284 -0.20519927  0.10189634]
mtx = np.loadtxt('outputs/camera_calibration/mtx.txt')  # cam.intrinsic  [-0.14574141 -0.20541     0.12561524]
# dist = np.zeros(5, dtype=mtx.dtype)  # np.zeros(5, dtype=mtx.dtype)
dist = np.loadtxt('outputs/camera_calibration/dist.txt')  # np.zeros(5, dtype=mtx.dtype)
base = []
for _ in range(50):
    radius = np.random.uniform(0.47, 0.67)
    cam.sample_a_pose_from_a_sphere(np.array([0, 0, 0]), radius)
    rgba, depth, mask = cam.get_camera_image()
    bgr = rgba[..., 0:3][..., ::-1]
    corners, ids, rejected_img_points = detect_markers(bgr, aruco_dict, board, board_params, mtx, dist)
    if ids is not None:
        rvec, tvec, marker_points = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)
        # rots = rodrigues_to_rotation_matrix(rvec)
        draw_frame = bgr.copy()
        aruco.drawDetectedMarkers(draw_frame, corners)
        for i in range(rvec.shape[0]):
            aruco.drawAxis(draw_frame, mtx, dist, rvec[i], tvec[i], 0.02)
        retval, b_rvec, b_tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist, rvec, tvec)
        p_w = np.dot(cam.pose, np.concatenate([b_tvec, np.ones((1, 1), dtype=b_tvec.dtype)], axis=0))
        base.append(p_w[0:3, 0])
        b_rot = rodrigues_to_rotation_matrix(b_rvec)
        aruco.drawAxis(draw_frame, mtx, dist, b_rvec, b_tvec, 0.2)
        cv2.imwrite('outputs/pose_estimation/{}_{}_{}.png'.format(cam.pose[0, 3], cam.pose[1, 3], cam.pose[2, 3]),
                    draw_frame)
base = np.stack(base, axis=0)
rms_error = np.linalg.norm(base-ground_truth_base.reshape(1, 3), axis=1)
avg_rms_error = np.mean(rms_error)
print('ground truth base position: \n{}'.format(ground_truth_base))
print('estimated average base position: \n{}'.format(np.mean(base, axis=0)))
print('RMS error for each estimated base positions: \n{}'.format(rms_error))
print('average RMS error for estimated base position: \n{}'.format(avg_rms_error))
