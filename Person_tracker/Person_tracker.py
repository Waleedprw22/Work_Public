import cv2
import csv
import collections
import numpy as np
from tracker import *
from pytube import YouTube
import time

# Initialize Tracker
tracker = EuclideanDistTracker()

# YouTube video URL
video_url = "https://www.youtube.com/watch?v=NyLF8nHIquM"

# Download the video
yt = YouTube(video_url)
video_stream = yt.streams.filter(file_extension='mp4', res='720p').first()
video_filename = video_stream.download()

# Initialize the videocapture object
cap = cv2.VideoCapture(video_filename)

input_size = 320

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Draw the rectangle
rectangle_color = (255, 0, 255)  # Magenta color
rectangle_thickness = 2


top_left_x = 100
top_left_y = 200
bottom_right_x = 200
bottom_right_y = 100

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')


# class index for our required detection classes
required_class_index = [0] # Focusing on the person class

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')
center_points_inside_rectangle = 0


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
top_left_x = 200
top_left_y = 200
bottom_right_x = 400
bottom_right_y = 400

temp_list = []
inner_list = []

# Function for count vehicle
# Initialize the previous state of the center point
prev_center_inside = False
center_points_entry_time = {}

# Function for counting the number of center points inside the rectangle
def count_person(box_id, img, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    global center_points_inside_rectangle, prev_center_inside, center_points_entry_time

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Check if the center point is inside the rectangle
    center_inside = (top_left_x <= ix <= bottom_right_x) and (top_left_y <= iy <= bottom_right_y)

    # Check for transition from outside to inside or vice versa
    if center_inside != prev_center_inside:
        if center_inside:
            # Center point entered the rectangle
            center_points_inside_rectangle += 1
            center_points_entry_time[id] = time.time()

        else:
            # Center point left the rectangle
            center_points_inside_rectangle -= 1
            if id in center_points_entry_time:
                entry_time = center_points_entry_time[id]
                exit_time = time.time()
                time_inside_rectangle = exit_time - entry_time
                print(f"Center point {id} stayed inside the rectangle for {time_inside_rectangle:.2f} seconds")
                del center_points_entry_time[id]  # Remove the entry from the dictionary
            
            
    prev_center_inside = center_inside

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)


# Function for finding the detected objects from the network output
def postProcess(outputs,img, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    indices = np.array(indices)

    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)

        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_person(box_id, img, top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        # Feed data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        postProcess(outputs,img, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    
        # Draw the rectangle on the image
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), rectangle_color, rectangle_thickness)

        # Draw the counting texts in the frame
        cv2.putText(img, "Person:        "+str(center_points_inside_rectangle)+"     ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break
    

    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realTime()
