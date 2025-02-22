import face_recognition
import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
import pyfirmata
import mysql.connector
import pickle

# Initialize face detector
detector = FaceDetector()

# Initial servo positions
servoPos = [180, 180]

# Arduino setup
port = "COM3"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')  # Pin 9 Arduino
servo_pinY = board.get_pin('d:10:s')  # Pin 10 Arduino

# Database setup
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="",  # Replace with your MySQL password
    database="face_recognition_db"
)
cursor = db.cursor()

# Retrieve known face images and names from the database
cursor.execute("SELECT name, image FROM known_faces")
rows = cursor.fetchall()
known_face_encodings = []
known_face_names = []

for row in rows:
    name, image_blob = row
    image = np.frombuffer(image_blob, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Close the cursor and connection for now (we'll reopen when needed)
cursor.close()
db.close()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
ws, hs = 1280, 720
video_capture.set(3, ws)
video_capture.set(4, hs)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame to save time
    if process_this_frame:
        # Resize frame to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color to RGB color
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if name != "Unknown":
            frame, bboxs = detector.findFaces(frame, draw=False)
            if bboxs:
                # Get the coordinate
                fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
                pos = [fx, fy]
                
                # Convert coordinate to servo degree
                servoX = np.interp(fx, [0, ws], [0, 360])
                servoY = np.interp(fy, [0, hs], [0, 360])

                servoX = min(max(servoX, 0), 360)
                servoY = min(max(servoY, 0), 360)

                servoPos[0] = servoX
                servoPos[1] = servoY

                # Draw target indicators
                cv2.circle(frame, (fx, fy), 80, (0, 0, 255), 2)
                cv2.putText(frame, str(pos), (fx + 15, fy - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.line(frame, (0, fy), (ws, fy), (0, 0, 0), 2)
                cv2.line(frame, (fx, hs), (fx, 0), (0, 0, 0), 2)
                cv2.circle(frame, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            else:
                cv2.putText(frame, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.circle(frame, (640, 360), 80, (0, 0, 255), 2)
                cv2.circle(frame, (640, 360), 15, (0, 0, 255), cv2.FILLED)
                cv2.line(frame, (0, 360), (ws, 360), (0, 0, 0), 2)
                cv2.line(frame, (640, hs), (640, 0), (0, 0, 0), 2)

    # Move servos
    servo_pinX.write(servoPos[0])
    servo_pinY.write(servoPos[1])

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()