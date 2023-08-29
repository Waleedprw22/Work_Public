# Interactive Presentation Tool

This project focuses on utilizing hand gesture recognition with voice control to either go to the next slide or the previous. I did run into some limitations that prevented 
me from maximizing the experiments that I could have performed, such as a broken webcam. Nonetheless, I performed a variety of different experiments. I removed the lines of code
that took frames from the webcam and replaced it with a list of two images: a thumbs down image and thumbs up image. I then randomized the selection of one of those images to prompt
the machine to either go to the next slide or to go back, updating the slide counter as we went along. I then incorporated voice recognition to also serve as a means to trigger a thumbs up image
or thumbs down image, changing the slide counter. In the voice recognition function, I provided a time frame in which the user has to speak so that there can be no unnecessary delays.

## Features

- Control slides with Hand Gestures: Use simple hand gestures like thumbs up and thumbs down to move to the next or previous slide.
- Voice Recognition: Trigger slide navigation by speaking voice commands like "next slide" or "previous slide."
- Randomized Slide Selection: Each slide displays a random image (thumbs up or thumbs down) for gesture recognition practice.
- Real-time Gesture and Voice Recognition: See real-time feedback on recognized gestures and voice commands.
- Easy to Use: Simply run the program and interact with the application using your microphone.

## Getting Started

To run this project on your local machine, follow these steps:

1. Obtain files for the original source code: https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/.
2. Obtain the images from this file.
3. Install the required dependencies
4. Replace the main python code with the file uploaded here.



## Usage

1. Run the program and ensure your microphone is functional.
2. Follow the on-screen instructions to speak voice commands (e.g., "next slide") or let the machine randomly generate a gesture based image.
3. The application will respond to recognized gestures and voice commands, allowing you to navigate through slides.



## Acknowledgements

- Source code from: https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/



