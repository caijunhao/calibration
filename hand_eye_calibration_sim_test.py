from utils import *
from simulator import RigidObject, Camera
from scipy.spatial.transform import Rotation as R
import numpy as np
import quaternion
import pybullet as p
import pybullet_data
import argparse
import json
import os


def pre_selection(ma, mb, scalar_th, angle_th):
    """
    Pre-selection includes
    1) selecting motions that the difference between the scalar values of hand and eye motions is small, and
    2) filter out the motion pairs in which the rotation axes are similar to the chosen candidates.
    :param ma: An Nx4x4-d numpy array including N camera motions.
    :param mb: An Nx4x4-d numpy array including N hand motions.
    :param scalar_th: the scalar threshold, see argparse for more detail.
    :param angle_th: the angle threshold, see argparse for more detail.
    :return: the selected camera and hand motions.
    """
    print('{} motions in total before pre-selection.'.format(ma.shape[0]))
    qa = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
    qb = quaternion.from_rotation_matrix(mb[:, 0:3, 0:3])
    diff_scalar_ab = np.abs(quaternion.as_float_array(qa)[:, 0] - quaternion.as_float_array(qb)[:, 0])
    flag_ab = diff_scalar_ab < scalar_th
    ma, mb = ma[flag_ab], mb[flag_ab]
    print('{} motions remained after removing motions with large scalar differences.'.format(ma.shape[0]))
    num_motion = ma.shape[0]
    assert num_motion > 0
    qa = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
    rot_vec_a = quaternion.as_rotation_vector(qa)
    rot_vec_a = rot_vec_a / np.linalg.norm(rot_vec_a, axis=1, keepdims=True)
    # pre-selection, remove similar motions
    max_discard_angle = np.deg2rad(angle_th)
    cos = np.cos(max_discard_angle)
    similarity_matrix = np.matmul(rot_vec_a, rot_vec_a.T)
    selection_matrix = np.logical_and(similarity_matrix >= -cos, similarity_matrix <= cos)
    selected_motion_ids = []
    remaining_ids = np.arange(num_motion)
    while selection_matrix.shape[0] != 0:
        selected_motion_ids.append(remaining_ids[0])
        distinct_flag = selection_matrix[0]
        remaining_ids = remaining_ids[distinct_flag]
        selection_matrix = selection_matrix[distinct_flag][:, distinct_flag]
    selected_motion_ids = np.array(selected_motion_ids)
    ma, mb = ma[selected_motion_ids], mb[selected_motion_ids]
    print('{} motions are selected for pose estimation.'.format(ma.shape[0]))
    return ma, mb


def dual_quaternion_method(ma, mb):
    """
    The hand-eye calibration method based on dual quaternions.
    This implementation is based on "Hand-Eye Calibration Using Dual Quaternions" by Konstantinos Daniilidis
    :param ma: An Nx4x4-d numpy array including N camera motions.
    :param mb: An Nx4x4-d numpy array including N hand motions.
    :return: A 4x4-d numpy array representing the pose from camera to gripper.
    """
    n = ma.shape[0]
    ta, tb = np.zeros((n, 4)), np.zeros((n, 4))
    ta[:, 1:], tb[:, 1:] = ma[:, 0:3, 3], mb[:, 0:3, 3]
    ta, tb = quaternion.from_float_array(ta), quaternion.from_float_array(tb)
    quat_a = quaternion.from_rotation_matrix(ma[:, 0:3, 0:3])
    quat_b = quaternion.from_rotation_matrix(mb[:, 0:3, 0:3])
    quat_a_p = 0.5 * ta * quat_a
    quat_b_p = 0.5 * tb * quat_b
    vec_a = quaternion.as_float_array(quat_a)[:, 1:]  # N * 3
    vec_a_p = quaternion.as_float_array(quat_a_p)[:, 1:]
    vec_b = quaternion.as_float_array(quat_b)[:, 1:]
    vec_b_p = quaternion.as_float_array(quat_b_p)[:, 1:]
    skew_ab = skew(vec_a+vec_b)  # N * 3 * 3
    skew_ab_p = skew(vec_a_p+vec_b_p)  # N * 3 * 3
    zero_vec, zero_mat = np.zeros_like(vec_a), np.zeros_like(skew_ab)
    eq1 = np.concatenate([(vec_a - vec_b).reshape(n, 3, 1),
                          skew_ab,
                          zero_vec.reshape(n, 3, 1),
                          zero_mat], axis=2).reshape(-1, 8)
    eq2 = np.concatenate([(vec_a_p - vec_b_p).reshape(n, 3, 1),
                          skew_ab_p,
                          (vec_a - vec_b).reshape(n, 3, 1),
                          skew_ab], axis=2).reshape(-1, 8)
    eqs = np.concatenate([eq1, eq2], axis=0)
    _, s, vh = np.linalg.svd(eqs)
    print('the singular values of the equation matrix: {}'.format(s))
    if s[-1] > 0.01 or s[-2] > 0.01:
        print('WARNING!!! The last two singular values are too large, the estimated result might be erroneous!')
    v7, v8 = vh[-1], vh[-2]
    u1, v1, u2, v2 = v7[:4], v7[4:], v8[:4], v8[4:]
    # s^2 * u1v1 + s * (u1v2+u2v1) + u2v2 = 0
    a = u1 @ v1
    b = u1 @ v2 + u2 @ v1
    c = u2 @ v2
    discriminant = b * b - 4 * a * c
    s1 = (-b + np.sqrt(discriminant)) / (2 * a)
    s2 = (-b - np.sqrt(discriminant)) / (2 * a)
    x1 = s1 ** 2 * u1 @ u1 + 2 * s1 * u1 @ u2 + u2 @ u2
    x2 = s2 ** 2 * u1 @ u1 + 2 * s2 * u1 @ u2 + u2 @ u2
    (s, x) = (s1, x1) if x1 > x2 else (s2, x2)
    lambda2 = np.sqrt(1 / x)
    lambda1 = s * lambda2
    q_check = lambda1 * v7 + lambda2 * v8
    q, q_p = quaternion.from_float_array(q_check[:4]), quaternion.from_float_array(q_check[4:])
    t_c2t = 2 * q_p * q.conj()
    rot_c2t = quaternion.as_rotation_matrix(q)
    t_c2t = quaternion.as_float_array(t_c2t)[1:]
    p_c2t = np.eye(4)
    p_c2t[0:3, 0:3] = rot_c2t
    p_c2t[0:3, 3] = t_c2t
    return p_c2t


def calibrate(args):
    # load aruco board configuration
    board_name = args.board_name
    dict_str = '_'.join(board_name.split('_')[1:4])
    col, row, length, sep, start_id = map(lambda x: int(x), board_name.split('_')[4:])
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT[dict_str])
    board = aruco.GridBoard_create(col, row, length / 1000, sep / 1000, aruco_dict, start_id)
    board_params = aruco.DetectorParameters_create()
    # set up pybullet environment
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF('plane.urdf')
    vis_params, col_params, body_params = get_param_template()
    col_params['fileName'] = vis_params['fileName'] = os.path.join('assets', board_name + '.obj')
    board_obj = RigidObject(board_name, vis_params, col_params, body_params)
    board_obj.set_pose_from_pos_and_quat([0, 0, 0.1], [0, 0, 0, 1])
    cam = Camera(**cfg['camera'])
    if os.path.exists('outputs/camera_calibration/mtx.txt'):
        mtx = np.loadtxt('outputs/camera_calibration/mtx.txt')
        dist = np.loadtxt('outputs/camera_calibration/dist.txt')
    else:
        mtx = cam.intrinsic
        dist = np.zeros(5, dtype=mtx.dtype)
    # create a virtual hand pose relative to the camera, use it as ground truth
    r, _, _ = np.linalg.svd(np.random.rand(3).reshape(3, 1))
    t = np.random.uniform(0.0, 0.077, 3)
    # determine the pose of tool center point (TCP) frame w.r.t. camera frame,
    # or say the transformation from camera to TCP
    p_c2t = np.eye(4)
    p_c2t[0:3, 0:3] = r
    p_c2t[0:3, 3] = t
    # collect camera and TCP motions
    i = 0
    p_c2w_e = []  # list of poses from camera to world Estimated by detecting aruco board
    p_b2t_g = []  # list of Ground truth poses from base to TCP
    p_b2t_n = []  # list of poses from base to TCP with Noise added into the ground truth poses
    while i < args.num_frame:
        radius = np.random.uniform(0.47, 0.67)
        cam.sample_a_pose_from_a_sphere(np.array([0, 0, 0]), radius)
        curr_p_b2c = cam.pose  # current pose of camera w.r.t. base frame
        curr_p_b2t = curr_p_b2c @ p_c2t  # current ground truth pose of TCP w.r.t. base
        noise_p_b2t = curr_p_b2t.copy()
        noise_p_b2t[0:3, 3] += np.random.normal(0.0, 0.001, 3)  # N(0, 0.001^2)
        axis = np.random.uniform(-1, 1, 3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-np.deg2rad(0.1), -np.deg2rad(0.1))
        delta_quat = np.zeros(4)
        delta_quat[0:3], delta_quat[3] = np.sin(angle / 2) * axis, np.cos(angle / 2)
        delta_rot = R.from_quat(delta_quat).as_matrix()  # !!! in the order of x, y, z, w
        noise_p_b2t[0:3, 0:3] = delta_rot @ noise_p_b2t[0:3, 0:3]
        p_b2t_g.append(curr_p_b2t)
        p_b2t_n.append(noise_p_b2t)
        rgba, depth, mask = cam.get_camera_image()
        bgr = rgba[..., 0:3][..., ::-1]
        corners, ids, rejected_img_points = detect_markers(bgr, aruco_dict, board, board_params, mtx, dist)
        if ids is not None:
            rvec, tvec, marker_points = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)
            # rots = rodrigues_to_rotation_matrix(rvec)
            draw_frame = bgr.copy()
            aruco.drawDetectedMarkers(draw_frame, corners)
            for j in range(rvec.shape[0]):
                aruco.drawAxis(draw_frame, mtx, dist, rvec[j], tvec[j], 0.02)
            retval, b_rvec, b_tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist, rvec, tvec)
            t_c2w = b_tvec[:, 0]
            rot_c2w = rodrigues_to_rotation_matrix(b_rvec)
            curr_p_c2w = np.eye(4)
            curr_p_c2w[0:3, 0:3], curr_p_c2w[0:3, 3] = rot_c2w, t_c2w
            p_c2w_e.append(curr_p_c2w)
            aruco.drawAxis(draw_frame, mtx, dist, b_rvec, b_tvec, 0.2)
            cv2.imwrite('outputs/hand_eye_calibration/{:04d}.png'.format(i), draw_frame)
            i += 1
    # motion selection
    p_c2w_e = np.stack(p_c2w_e, axis=0)
    p_b2t_g = np.stack(p_b2t_g, axis=0)
    p_b2t_n = np.stack(p_b2t_n, axis=0)
    frame_ids = np.array([[i, j] for i in range(args.num_frame - 1) for j in range(i + 1, args.num_frame)])
    ai, aj = p_c2w_e[frame_ids[:, 0]], p_c2w_e[frame_ids[:, 1]]
    bi, bj = p_b2t_g[frame_ids[:, 0]], p_b2t_g[frame_ids[:, 1]]
    bni, bnj = p_b2t_n[frame_ids[:, 0]], p_b2t_n[frame_ids[:, 1]]
    ma = np.matmul(ai, np.linalg.inv(aj))  # candidate camera motions
    mb = np.matmul(np.linalg.inv(bi), bj)
    mbn = np.matmul(np.linalg.inv(bni), bnj)  # candidate TCP motions with noise
    sma, smb = pre_selection(ma, mb, args.sc, args.a)
    sman, smbn = pre_selection(ma, mbn, args.sc, args.a)
    p_c2t_e = dual_quaternion_method(sma, smb)
    p_c2t_en = dual_quaternion_method(sman, smbn)
    print('estimated hand-eye pose from noise-free hand-motion data: \n{}'.format(p_c2t_e))
    print('estimated hand-eye pose from noise hand-motion data: \n{}'.format(p_c2t_en))
    print('ground truth hand-eye pose: \n{}'.format(p_c2t))
    # q_e = quaternion.as_float_array(quaternion.from_rotation_matrix(p_c2t_e))
    # q_en = quaternion.as_float_array(quaternion.from_rotation_matrix(p_c2t_en))
    # q = quaternion.as_float_array(quaternion.from_rotation_matrix(p_c2t))
    # rms_q_e = np.linalg.norm(q-q_e)
    # rms_q_en = np.linalg.norm(q-q_en)
    frob_norm_e = np.linalg.norm(p_c2t[0:3, 0:3]-p_c2t_e[0:3, 0:3])
    frob_norm_en = np.linalg.norm(p_c2t[0:3, 0:3]-p_c2t_en[0:3, 0:3])
    rms_t_e = np.linalg.norm(p_c2t[0:3, 3]-p_c2t_e[0:3, 3])
    rms_t_en = np.linalg.norm(p_c2t[0:3, 3]-p_c2t_en[0:3, 3])
    print('Frobenius norm of the difference of rotation matrices for noise-free hand-motion data: {}'.format(frob_norm_e))
    print('RMS of the errors in translation for noise-free hand-motion data: {}'.format(rms_t_e))
    print('Frobenius norm of the difference of rotation matrices for noise hand-motion data: {}'.format(frob_norm_en))
    print('RMS of the errors in translation for noise hand-motion data: {}'.format(rms_t_en))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand-eye calibration in PyBullet.')
    parser.add_argument('--board_name',
                        type=str,
                        default='ArUco_DICT_7X7_50_5_7_50_10_0',
                        help='the name of aruco board.')
    parser.add_argument('--num_frame',
                        type=int,
                        default=52,
                        help='The required number of frames captured from the camera.')
    parser.add_argument('--a',
                        type=int,
                        default=15,
                        help='one of the motions for which '
                             'the angle of rotation axes is in [a, pi-a] will be discarded.')
    parser.add_argument('--sc',
                        type=float,
                        default=0.0002,
                        help='the scalar threshold for the difference '
                             'between two scalar parts of the quaternions of hand and eye motions.')
    calibrate(parser.parse_args())
