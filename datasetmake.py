# Import OpenCV2 
import cv2

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id

# Get the name
name = raw_input('Enter your name: ')

# Initialize sample face image
count = 0

global Id

#storing data in file
with open("metadata.txt") as input:
    # Read non-empty lines from input file
    lines = [line for line in input if line.strip()]
with open("metadata.txt", "a") as output:
	Id = int(lines[-1].split()[0])+1 # Get the Id from the file
	output.write("%d %s\n"%(Id,name)) # Add entry to the file
	
while(True):

    #Capture video
    _, image_frame = vid_cam.read()

    #Convert to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    #Detect frames 
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    #for each face
    for (x,y,w,h) in faces:

        # Convert the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("C:/Users/Rahul/datasets/User." + str(Id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame
        cv2.imshow('frame', image_frame)


    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image=100, stop 
    elif count>99:
        break

#Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
