"""tracker_demo.py

This code explain the concept behind tracker in easiest way, 
so the performance and code style might not clean as you expected.

KEYWORD: nested loops, Python, OpenCV, object detection

Reference
Pysource: Object Tracking from scratch with OpenCV and Python
https://www.youtube.com/watch?v=GgGro5IV-cs

Last but not least, big thank to my mentor MaxChang during internship

Feel free if you have any question to ask :)
"""

import sys
import math
import string

import cv2


RED = (0, 0, 255) # bgr
DISTANCE_LIMIT = 50 # NOTE: This number change dramatically with different video input

def detectModel(cam):
    """AI model, such as YOLO.
    
    Assume you already have one LOL

    Arguments:
        cam: video or RTSP
        
    Return:
        boxes: bounding box with (class_id, confidence_score, bboxes),
        bboxes store the exact position of topleft and lowerright,
        or width, height and topleft position
    """
    boxes = None
    return boxes

def open_window(win_name:string, title:string):
    """Open the display window.
    
    Arguments:
        win_name: it won't dispaly on screen
        title: display on the top of the window
    
    Return:
        OpenCV window's display
    
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(win_name, title)


win_name    = "Window 01"
title       = "Test tracker"

cam = cv2.videoCapture("url")
open_window(win_name, title)
boxes = detectModel(cam)

frame_count     = 0
Track_id        = 0
Tracker_object  = {}
 

while True:
    _, frame=cam.read()

    cur_pnt = [] # current point, initialize it every frame
    frame_count += 1
    
    print("="*10)
    print("Frame: {}".format(frame_count))


    for idx, bb in enumerate(boxes):
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        center_x, center_y = int((x_min + x_max)/2), int((y_min + y_max)/2)
        # draw bboxes
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), RED, thickness=2)
        # save current point position
        cur_pnt.append((center_x, center_y))
        
    # First frame 
    # since it cant not refer to previous point 
    if frame_count <= 1:
        """Add every point to Tracker_object"""

        for pt in cur_pnt:
            Tracker_object[Track_id] = pt
            Track_id +=1

    # Following
    else:
        """Mark it with same id if distance is close enough"""
        
        # list and dictionary can't change during loop
        Tracker_object_copy = Tracker_object.copy()
        cur_pnt_copy        = cur_pnt.copy() 

        for id, pt2 in enumerate(Tracker_object_copy):
            
            # initialize condition with boolean
            object_exist = False
            
            for pt in cur_pnt_copy:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                
                # update new position with same id
                if distance < DISTANCE_LIMIT:
                    
                    Tracker_object[id] = pt
                    object_exist = True
                    # remove the remaining points
                    # we find what we need, so the following calculation is not required
                    cur_pnt.clear()
                    break
            if not object_exist:
                """Delete id if miss the object position"""

                print("Out of bound with distance: {} position: {}".format(distance, pt))
                Tracker_object.pop(id)
        
        for pt in cur_pnt:
            """Remaining points are NEW object"""
            
            Tracker_object[Track_id] = pt
            Track_id +=1

    # Display label on screen
    for id, pt in enumerate(Tracker_object):
            cv2.circle(frame, pt, 5, RED, -1)
            cv2.putText(frame, str(id), (pt[0], pt[1]-7), 0, 1, RED, 2)
    
    print("Tracker_object")
    print(Tracker_object)
    
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
sys.exit()