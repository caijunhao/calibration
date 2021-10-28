from simulator import Camera, RigidObject
from utils import *
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import argparse
import json
import os


parser = argparse.ArgumentParser(description='camera calibration')
parser.add_argument('--num_img',
                    type=int,
                    default=50,
                    help='number of images used for calibration')
parser.add_argument('--height',
                    type=int,
                    default=7,
                    help='height of the chessboard')
parser.add_argument('--width',
                    type=int,
                    default=9,
                    help='width of the chessboard')
parser.add_argument('--size',
                    type=float,
                    default=0.03,
                    help='square size for each square of the chessboard')
args = parser.parse_args()


def find_chessboard_corners(bgr, width, height, criteria):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
    if ret:
        # corners = np.squeeze(corners, axis=1)
        img_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, np.squeeze(corners, axis=1), np.squeeze(img_corners, axis=1).astype(np.float32)
    else:
        return ret, corners, None


def calibrate():
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF('plane.urdf')
    vis_params, col_params, body_params = get_param_template()
    board_name = 'chessboard_8_10_30_15'
    col_params['fileName'] = vis_params['fileName'] = os.path.join('assets', board_name + '.obj')
    board_obj = RigidObject(board_name, vis_params, col_params, body_params)
    board_obj.set_pose_from_pos_and_quat([0, 0, 0.1], [0, 0, 0, 1])
    cam = Camera(**cfg['camera'])

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_pts = np.zeros((args.height * args.width, 3), dtype=np.float32)
    corner_pts[:, :2] = np.mgrid[0:args.width, 0:args.height].T.reshape(-1, 2)
    corner_pts = corner_pts * args.size
    corner_pts_list = list()
    img_pts_list = list()
    img_h, img_w = cam.height, cam.width

    i = 0
    while i < args.num_img:
        # capturing chessboard with various distance will achieve more precise estimation
        radius = np.random.uniform(0.47, 0.67)
        cam.sample_a_pose_from_a_sphere(np.array([0, 0, 0]), radius)
        rgba, depth, mask = cam.get_camera_image()
        bgr = rgba[..., 0:3][..., ::-1]
        ret, corners, img_corners = find_chessboard_corners(bgr, args.width, args.height, criteria)
        if ret:
            print('chessboard detection succeeded.')
            img = cv2.drawChessboardCorners(bgr.copy(), (args.width, args.height), img_corners, ret)
            cv2.imwrite('outputs/camera_calibration/{:04d}_color.png'.format(i), bgr)
            cv2.imwrite('outputs/camera_calibration/{:04d}_depth.png'.format(i), depth)
            cv2.imwrite('outputs/camera_calibration/{:04d}_display.png'.format(i), img)
            corner_pts_list.append(corner_pts)
            img_pts_list.append(img_corners)
            i += 1
        else:
            print('chessboard detection failed, please try other viewpoints ... ')
            continue
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corner_pts_list, img_pts_list, (img_w, img_h), None, None)
    if ret:
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
        print('Ground truth cameraMatrix: {}'.format(cam.intrinsic))
        print('cameraMatrix: {}'.format(mtx))
        print('distCoeffs: {}'.format(dist))
        print('calibrated intrinsic: {}'.format(new_mtx))
        np.savetxt('outputs/camera_calibration/mtx.txt', mtx)
        np.savetxt('outputs/camera_calibration/dist.txt', dist)
        np.savetxt('outputs/camera_calibration/new_mtx.txt', new_mtx)
    else:
        print('fail to calibrate the camera, exit and try again.')
        exit()


if __name__ == '__main__':
    calibrate()


