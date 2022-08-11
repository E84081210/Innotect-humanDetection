# Peopleflow counting with YOLOv4-TensorRT
-----------------

Example demonstrates people-flow counting with YOLOv4 model and TensorRT speedup. Notice that it's only work on NVIDIA Jetson Platform.

Features:
- AreaDetection
    - Target    : user define which area should be focus on, i.e. where counter operate.
    - Ignore    : bounding boxes won't show up or participate further computation.
- Counter
    - Exit & Entrance : classify whether person is getting out or in the area.
    - FlowCount       : total of flow count so far.
- Application(optional)
    - Once people enter area, you can do things like: sent email to host, open door, alert on screen, etc.

Tech behind it
- Tracker
    We will track object via the chage of central position, instead of using tracker from OpenCV, which is well optimized. 
    I also wrote a simple code about foundation of tracker, check **tracker_demo.py** if you are interested.
- Object detecion with YOLOv4
    Tracker needs the exact position of object, which means we have to recognize object first.
- Speedup with NVIDIA TensorRT
    It's not a simple lesson at all, but I recommend take a look at NVIDIA official website.

# Table of contents
-----------------
