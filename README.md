# Peopleflow counting with YOLOv4-TensorRT

Example demonstrates people-flow counting with YOLOv4 model with TensorRT.<br>
Notice that it specifically works on NVIDIA Jetson Platform.

#### Features:
- AreaDetection
    - Target    : user define which area should be focus on, i.e. where counter operate.
    - Ignore    : bounding boxes won't show up or participate further computation.
- Counter
    - Exit & Entrance : classify whether person is getting out or in the area.
    - FlowCount       : total of flow count so far.
- Docker
    - Build environment faster.  
- Application(optional)
    - Once people enter area, you can do things like: sent email to host, open door, alert on screen, etc.
#### Tech behind it
- Tracker<br>
    We will track object via the chage of central position, instead of using tracker from OpenCV, which is well optimized.<br>
    I also wrote a simple code about foundation of tracker, check **tracker_demo.py** if you are interested.
- Object detecion with YOLOv4<br>
    Tracker needs the exact position of object, which means we have to get information of bounding box first.
- Speedup with NVIDIA TensorRT<br>
    It's not simple at all, but I recommend take a look at [NVIDIA official website](https://developer.nvidia.com/tensorrt).

# Table of contents

