# Instance Segmentation and Object Pose Estimation Project

Welcome to the Instance Segmentation and Object Pose Estimation project! In this project, we tackle the challenges of instance segmentation and estimating the pose of objects using RGB images and depth information.

## Project Outline

I first generated my dataset which was fed into a mini U-net model instance I then developed. Before feeding data into the CNN, it was important to create a structured Dataset
that acts as a mapping between integer indices and the corresponding data samples. Additionally, a DataLoader was necessary to efficiently 
iterate through the dataset, group the data into batches, and optionally shuffle the training set. However, it's important 
to note that shuffling should be avoided for the validation and test sets to ensure consistent evaluation and comparison of 
results.After getting an mIOU of around 94% on the train data and around 88% on the validation data, I saved the predicted masks that were produced.


![Figure 1: Unet architecture](https://imgur.com/a/eBouwYJ))

![Figure 2: mIOU vs Epoch graph](https://imgur.com/a/aEeCF7d))

I then prepared two point clouds: one  down-sampled from the .obj file of the object and the other, which was projected from 
 the depth image of the object. Following this, I used Iterative Closest Point (ICP) to align the two point clouds, outputting a transformation that
turns the sampled cloud to the projected one. Following this, I conducted object pose estimation with quite a high accuracy and have made sure to 
visualize the results.


![Figure 3: Results Visualized](https://imgur.com/a/wZvE4ME))

## Features

- Implementation of a neural network for instance segmentation of objects in RGB images.
- Utilization of depth information for improved object pose estimation.
- Generation of training data and dataset preprocessing.
- Evaluation of segmentation and pose estimation results.
- Visualization of segmented objects and their estimated poses.



 
