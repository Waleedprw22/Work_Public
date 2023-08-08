# Center Point Tracking Project

This project focuses on tracking people, through center points, detected in videos using YOLOv3 and OpenCV. It counts the number of center points that enter or leave a specified rectangular region and calculates the time each center point spends inside the region.
The original source code was centered around simply detecting vehicles that either go up or down, which is applicable for one specific video. I changed the class to focus on
identifying people instead. I also removed the lines and developed a random rectangular region of interest in the middle of the video. I adjusted the original "count_vehicle" function
into a "count_person" function that has a live of counter of how many people are in the region of interest per frame. Furthermore, I also added a feature that can inform you how long
each person spends within the region. 
## Features

- Real-time video processing using YOLOv3 and OpenCV.
- Tracks center points of people.
- Counts center points entering or leaving a designated rectangular region.
- Calculates and displays time spent by each center point inside the region.

## Getting Started

To run this project on your local machine, follow these steps:

1. Clone this repository: `git clone https://github.com/yourusername/center-point-tracking.git`
2. Install the required dependencies: `pip install opencv-python pytube`
3. Download the YOLOv3 model files and place them in the appropriate directory.
4. Run the script: `python center_point_timer.py`

Make sure to customize the region coordinates and other settings according to your requirements.

## Dependencies

- OpenCV: `pip install opencv-python`
- pytube: `pip install pytube`

## Usage

- Adjust the `top_left_x`, `top_left_y`, `bottom_right_x`, and `bottom_right_y` variables in the `center_point_timer.py` script to define the rectangular region of interest.
- Run the script using a Python interpreter. The video feed will open, and you'll see the center points being tracked along with their stay duration in the region.



## Acknowledgements

- YOLOv3: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- OpenCV: [Open Source Computer Vision Library](https://opencv.org/)
- Source code from: https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/


