# Import OpenCV2
import cv2
# Import os for file path
import os

# Import numpy 
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.createLBPHFaceRecognizer()

# Using prebuilt frontal face training model
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create function to get the images and ID data
def getImages_ids(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImages_ids('C:/Users/Rahul/datasets')

# Train the model
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.save('C:/Users/Rahul/trainer/trainer.yml')
