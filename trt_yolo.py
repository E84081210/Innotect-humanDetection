"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import math
import os, sys, logging
import time
import argparse
import subprocess

import cv2
import numpy as np
import pycuda.autoinit
#from nvidia.Desktop.tensorrt_demos.utils import AreaDetection  # This is needed for initializing CUDA driver

from utils.style import BLUE, RED, GREEN, WHITE, KELIN, BLACK, SLIVER, \
    FONT_SCALE, FONT, LINE, ALPHA, Remind
from utils.area_detect import AreaDetection
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME     = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.5,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.
    
    # Arguments
        cam: the camera instance (video source).
        trt_yolo: the TRT YOLO object detector instance.
        conf_th: confidence/score threshold for object detection.
        vis: for visualization.
    """
    distance_limit  = cam.img_height/12

    full_scrn       = False
    fps             = 0.0
    frame_count     = 0
    Track_object    = {}
    Track_id        = 0 
    FlowCount       = 0
    FlowUp          = 0
    FlowDown        = 0
    
    # draw polygon area
    win1, win2 = "Area01", "Area02"
    win_name1, win_name2 = "Draw Area 01", "Draw Area 02"
    area1 = AreaDetection(cam, win1, win_name1)
    area2 = AreaDetection(cam, win2, win_name2, area1)
    
    while True:
        frame_count += 1
        
        # initialize current point
        # [center_x, center_y, situation, history]
        # situation   = 1     if object is in area, passing cv2.pointPolygonTest
        # history     = 1     if object has never entered area, i.e. never been counted
        cur_pnt = []

        tic = time.time()
        img = cam.read()
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0 or img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        # display drawn area
        overlay1, overlay2 = img.copy(), img.copy()
        cv2.fillPoly(overlay1, pts=[np.array(area1.save_pnt, np.int32)], color=KELIN)
        cv2.fillPoly(overlay2, pts=[np.array(area2.save_pnt, np.int32)], color=SLIVER)
        img = cv2.addWeighted(img, 0.7, overlay1, 0.3, 0)
        img = cv2.addWeighted(img, 0.8, overlay2, 0.2, 0)

        for idx, bb in enumerate(boxes):     

            if (clss[idx]!=0):
                continue

            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            center_x, center_y = int((x_min + x_max)/2), int((y_min + y_max)/2)
            
            interest = cv2.pointPolygonTest(np.array(area1.save_pnt, np.int32), (center_x, int(y_max)), False)
            un_interset = cv2.pointPolygonTest(np.array(area2.save_pnt, np.int32), (center_x, int(y_max)), False)
            
            
            if un_interset == 1:
                continue    
            elif interest == 1:
                cur_pnt.append([center_x, center_y, 1, 1])
                # Open the door
                # MAKE SURE you have 100% confidence with your program
                # subprocess("mosquitto_pub -h 172.16.21.2 -t action -m OpenDoor", shell=True)
            else:
                cur_pnt.append([center_x, center_y, 0, 1])

        
        # First frame
        if frame_count == 1:
            for pt in cur_pnt:
                if pt[2] == 1:
                    pt[3] = 0
                Track_object[Track_id] = pt
                Track_id += 1

        # The following
        if frame_count > 1:
            # due to the rule -- you can't change list during loop
            Tracking_object_copy = Track_object.copy()
            cur_pnt_copy = cur_pnt.copy()
            for id, pre_pt in Tracking_object_copy.items():
                object_exist = False
                for pt in cur_pnt_copy:

                    # distance limit turn up when object get close too camera
                    if pt[1] <= (cam.img_height/4):
                        limitation = distance_limit
                    elif (cam.img_height/4) < pt[1] and pt[1] <= (cam.img_height * 2/4):
                        limitation = distance_limit*2
                    elif (cam.img_height * 2/4) < pt[1] and pt[1] <= (cam.img_height * 3/4):
                        limitation = distance_limit*3
                    else:
                        limitation = distance_limit*4
                    
                    dis_x, dis_y = pre_pt[0]-pt[0],pre_pt[1]-pt[1]
                    distance = math.hypot(dis_x, dis_y)
                    
                    # update id position
                    if distance < limitation:
                        
                        if pre_pt[3] == 1:
                            # not in area -> in area
                            if pre_pt[2] == 0 and pt[2] == 1:
                                FlowCount += 1
                                if dis_y > 0:
                                    FlowUp += 1
                                else:
                                    FlowDown += 1 
                                pt[3] = 0
                            else:
                                continue
                        
                        # remain history status
                        if pre_pt[3] == 0:
                            pt[3] = 0
                        
                        Track_object[id] = pt
                        object_exist = True

                        # if object is not exist, remain point
                        if pt in cur_pnt:
                            cur_pnt.remove(pt)
                        break
                    
                # pop the id if it can't be tracked
                if not object_exist:
                    Track_object.pop(id)

            # points remain -> new object
            for pt in cur_pnt:
                # NOTE
                # for RTSP, people don't show up without walking around beforehand
                # thus, those who appear unexpectedly should not be counted
                if pt[2] == 1:
                    pt[3] = 0
                Track_object[Track_id] = pt
                Track_id += 1

        
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        img = vis.draw_bboxes(img, boxes, confs, clss, ignoreArea=area2, focusArea=area1)
        img = show_fps(img, fps)
        Remind(img,
            "FlowCount  : {}".format(FlowCount), 
            "Exit         : {}".format(FlowUp),
            "Entrance   : {}".format(FlowDown))
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)                        

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)
    cam.release()
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == '__main__':
    main()