# Object-detection-in-real-time

## Table of Contents
+ [About](#about)
+ [Requirements](#installing)
    + [Environment](#env)
    + [OS Requirements](#osinstalling)
    + [Python Requirements](#pyinstalling)
+ [Running the code](#run)
+ [Output](#out)
+ [Result](#result)

## About <a name = "about"></a>

In this project, a real-time object detection system was implemented using the YOLOv4 model in Python with OpenCV and the Vidgear library for video streaming. The primary objective was to detect and count objects, specifically cars and persons, in a live video stream sourced from YouTube.

## Requirements <a name = "installing"></a>

This project has been tested on Ubuntu 20.04.

## Environment <a name = "env"></a>

The project was implemented in a local Python environment, leveraging OpenCV for computer vision tasks and Vidgear for handling the video stream.

### OS Requirements <a name = "osinstalling"></a>
Install the following library:
1. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

### Python Requirements <a name = "pyinstalling"></a>

1. Now you need to create a conda environment:
    ```ShellSession
    conda create -n myenv
    conda activate myenv
    ```
    
2. Install the required python packages
    ```ShellSession
    pip install vidgear
    pip install opencv-python
    pip install yt_dlp
    ```
    
## Running the code <a name = "run"></a>

In order to run the code, make sure you are in the correct virtual environment and run the python file:
```ShellSession
$ conda activate myenv
$ python3 object_detection.py
```

## Output <a name = "out"></a>
The processed video frames were displayed in real-time, showcasing detected objects with bounding boxes and corresponding counts.
- Bounding boxes were drawn around detected objects, and labels with confidence scores were added.
- The system counted the number of detected cars and persons in each frame, displaying these counts on the video.

## Result <a name = "result"></a>
The system successfully detected and counted cars and persons in a live video stream, demonstrating the effectiveness of the YOLOv4 model for real-time object detection.

![image](https://github.com/user-attachments/assets/99d4fca9-c220-4d44-912b-5f0237cb0c93)
