import pybullet as p
import numpy as np
import cv2
from cv2 import aruco


ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
}


def get_param_template():
    vis_params = {'shapeType': p.GEOM_MESH,
                  'fileName': '',
                  'meshScale': [1, 1, 1]}
    col_params = {'shapeType': p.GEOM_MESH,
                  'fileName': '',
                  'meshScale': [1, 1, 1]}
    body_params = {'baseMass': 0,
                   'basePosition': [0, 0, 0],
                   'baseOrientation': [0, 0, 0, 1]}
    return vis_params, col_params, body_params


def detect_markers(bgr, aruco_dict, board, params, mtx, dist):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                            parameters=params,
                                                            cameraMatrix=mtx,
                                                            distCoeff=dist)
    aruco.refineDetectedMarkers(gray, board, corners, ids, rejected_img_points)
    return corners, ids, rejected_img_points


def rodrigues_to_rotation_matrix(rvec):
    """
    :param rvec: Nx1x3-d or 3x1-d numpy array representing rotation vector(s)
    :return: Nx3x3-d or 3x3-d numpy array representing the corresponding rotation matrix
    """
    rvec = np.squeeze(rvec)
    if len(rvec.shape) == 1:
        rvec = np.expand_dims(rvec, axis=0)  # 1x3
    num_vec = rvec.shape[0]
    theta = np.linalg.norm(rvec, axis=1, keepdims=True)  # Nx1
    r = rvec / theta  # Nx3
    theta = theta.reshape((num_vec, 1, 1))
    zero_vec = np.zeros_like(r[:, 0])
    rot = np.cos(theta) * np.eye(3).reshape(1, 3, 3) +\
          (1 - np.cos(theta)) * r.reshape(num_vec, 3, 1) @ r.reshape(num_vec, 1, 3) +\
          np.sin(theta) * np.array([[zero_vec, -r[:, 2], r[:, 1]],
                                    [r[:, 2], zero_vec, -r[:, 0]],
                                    [-r[:, 1], r[:, 0], zero_vec]]).transpose((2, 0, 1))
    return rot


def skew(mat):
    """
    Copy from: https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array
    This function returns a numpy array with the skew symmetric cross product matrix for vector.
    The skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)
    :param mat: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """
    if mat.ndim == 1:
        return np.array([[0, -mat[2], mat[1]],
                         [mat[2], 0, -mat[0]],
                         [-mat[1], mat[0], 0]])
    else:
        shape = mat.shape[0:-1]
        zeros = np.zeros(shape, dtype=mat.dtype)
        skew_mat = np.stack([zeros, -mat[..., 2], mat[..., 1],
                             mat[..., 2], zeros, -mat[..., 0],
                             -mat[..., 1], mat[..., 0], zeros], axis=-1)
        new_shape = list(mat.shape) + [3]
        skew_mat = skew_mat.reshape(new_shape)
        return skew_mat
