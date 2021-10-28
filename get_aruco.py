from utils import ARUCO_DICT
import numpy as np
import cv2
from cv2 import aruco
import argparse
import os

parser = argparse.ArgumentParser(description='generate aruco marker or board')
parser.add_argument('--dict',
                    type=int,
                    default=7,
                    help='the type of the aruco dictionary, '
                         'only 4, 5, 6, and 7 are allowed in this code (default 7).')
parser.add_argument('--num_id',
                    type=int,
                    default=50,
                    help='the total number of unique aruco ids that can be generated in the specified dict, '
                         'only 50, 100, 250, 1000 are valid (default 50).')
parser.add_argument('--type',
                    type=str,
                    default='a',
                    help='m for single marker, a for aruco board, and c for charuco board')
parser.add_argument('--id',
                    type=int,
                    default=0,
                    help='the id of the marker to be generated, '
                         'should be less than num_id, for single marker generation only.')
parser.add_argument('--num_col',
                    type=int,
                    default=5,
                    help='the number of the aruco markers in a row of the board.')
parser.add_argument('--num_row',
                    type=int,
                    default=7,
                    help='the number of the aruco markers in a column of the board.')
parser.add_argument('--length',
                    type=float,
                    default=500,
                    help='marker side length, in millimeters.')
parser.add_argument('--sep',
                    type=float,
                    default=100,
                    help='the separation length between two markers, in millimeters.')
parser.add_argument('--margin',
                    type=int,
                    default=1,
                    help='the flag is used to specify whether to generate margin for marker or board or not.')
args = parser.parse_args()


if __name__ == '__main__':
    assert args.dict in [4, 5, 6, 7]
    assert args.num_id in [50, 100, 250, 1000]
    assert args.id < args.num_id
    if not os.path.exists('markers'):
        os.makedirs('markers')
    dict_str = 'DICT_{}X{}_{}'.format(args.dict, args.dict, args.num_id)
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT[dict_str])
    if args.type == 'a':
        assert args.num_row > 0 and args.num_col > 0
        assert args.id + args.num_row * args.num_col <= args.num_id
        board = aruco.GridBoard_create(args.num_col, args.num_row,
                                       markerLength=args.length/100,
                                       markerSeparation=args.sep/100,
                                       dictionary=aruco_dict,
                                       firstMarker=args.id)
        height = args.num_row * args.length + (args.num_row - 1) * args.sep
        width = args.num_col * args.length + (args.num_col - 1) * args.sep
        marker_img = board.draw((width, height))
        if args.margin:
            img = np.ones((height + args.sep * 4, width + args.sep * 4), dtype=marker_img.dtype) * 255
            # add white boundary to the board
            img[args.sep * 2:height + args.sep * 2, args.sep * 2:width + args.sep * 2] = marker_img
        else:
            img = marker_img
        cv2.imwrite('markers/ArUco_DICT_{}X{}_{}_{}_{}_{}_{}_{}.png'.format(args.dict, args.dict, args.num_id,
                                                                            args.num_col, args.num_row,
                                                                            args.length, args.sep,
                                                                            args.id), img)
    elif args.type == 'c':
        assert args.num_row > 0 and args.num_col > 0
        assert args.num_row * args.num_col <= args.num_id
        board = aruco.CharucoBoard_create(args.num_col, args.num_row,
                                          squareLength=(args.length+args.sep*2)/100,
                                          markerLength=args.length/100,
                                          dictionary=aruco_dict)
        height = args.num_row * (args.length+args.sep*2) + 4 * args.sep
        width = args.num_col * (args.length+args.sep*2) + 4 * args.sep
        marker_img = board.draw((width, height))
        if args.margin:
            img = np.ones((height + args.sep * 4, width + args.sep * 4), dtype=marker_img.dtype) * 255
            # add white boundary to the board
            img[args.sep * 2:height + args.sep * 2, args.sep * 2:width + args.sep * 2] = marker_img
        else:
            img = marker_img
        cv2.imwrite('markers/ChArUco_DICT_{}X{}_{}_{}_{}_{}_{}.png'.format(args.dict, args.dict, args.num_id,
                                                                           args.num_col, args.num_row,
                                                                           args.length, args.sep), img)
    elif args.type == 'm':
        marker_img = np.zeros((args.length, args.length), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, args.id, args.length, marker_img, 1)
        if args.margin:
            img = np.ones((args.length + args.sep * 2, args.length + args.sep * 2), dtype=marker_img.dtype) * 255
            # add white boundary to the board
            img[args.sep:args.length + args.sep, args.sep:args.length + args.sep] = marker_img
        else:
            img = marker_img
        cv2.imwrite('markers/DICT_{}X{}_{}_{}_{}.png'.format(args.dict, args.dict, args.num_id,
                                                             args.length,
                                                             args.id), img)


