import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import time
import speech_recognition as sr


# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

#Class Names
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

# List of image paths
image_paths = ['thumbs_up.png', 'thumbs_down.png']

current_slide = 1
recognizer = sr.Recognizer()

def recognize_voice():
    with sr.Microphone() as source:
        print("Say 'next slide' or 'previous slide'")
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            audio = recognizer.listen(source, timeout=5)  # Set a timeout of 5 seconds
            voice_text = recognizer.recognize_google(audio)
            print("Recognized voice:", voice_text)
            return voice_text.lower()
        except sr.WaitTimeoutError:
            print("No audio captured within the timeout")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        
while True:
    # Randomly select thumbs up or thumbs down image
    speech = recognize_voice()
    if speech == 'next slide':
        random_image_path = 'thumbs_up.png'
    elif speech == 'previous slide':
        random_image_path = 'thumbs_down.png'
    else:
        print('Voice not detected')
        random_image_path = random.choice(image_paths)
    
    # Load and preprocess the random image
    frame = cv2.imread(random_image_path)

    x, y, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    gesture_command = None

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            gesture_command = classNames[classID]


    if gesture_command:
        if gesture_command == 'thumbs up':
            current_slide += 1
            print("Next Slide")
            print(f"Slide: {current_slide}")

        elif gesture_command == 'thumbs down':
            current_slide -= 1
            if current_slide < 1:
                current_slide = 1
            f"Slide: {current_slide}"
            print("Previous Slide")

    cv2.putText(frame, f"Slide: {current_slide}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Presentation Tool", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
