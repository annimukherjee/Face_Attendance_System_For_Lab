#image size should be less than 150 kb and jpg format

import face_recognition        #library for face recognition
import cv2
import numpy as np
import time
import pandas as pd
import os
import tkinter as tk           # for defining mouse click functions
from datetime import datetime,timezone
from multiprocessing import Process
video_capture = cv2.VideoCapture(0)
# data0=[]
logging_file = open('log.txt',"a")
global window
global flag
flag=0
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = 1  # initialize counter outside the function

def mouse_click(event,x,y,flags,params):
    global time_type, counter
    time_type=""
    if event == cv2.EVENT_LBUTTONDOWN:
        if x<320:
            time_type="in_time"
        else:
            time_type="out_time"
        
        if counter % 2 == 0:  # write to log file only on even clicks
            # Process(target=insert_data,
            #     args=(time_type, name, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))).start()
            logging_file.write(name + " " + time_type + " " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        counter += 1  # increment the counter on every click
        flag=1
        window.destroy()

folderPath = 'Images'
known_face_encodings = []
known_face_names = []

for file in os.listdir(folderPath):
    # Load image and get face encoding
    img_path = os.path.join(folderPath, file)
    image = face_recognition.load_image_file(img_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    
    # Extract name from file name
    name = file.split('.')[0]
    
    # Append face encoding and name to lists
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

    

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# names =set()s
# prev_time = time.time()
# store_interval = 10

while True:
    if(flag==1):
        time.sleep(2)
        flag=0
    ret, frame = video_capture.read()
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video',mouse_click)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            name = "Unknown or try again...."
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            # prev=len(names)
            # names.add(name)
            # if len(names) > prev:
            #     t = time.localtime()
            #     current_time = time.strftime("%H:%M:%S", t)
            #     data.append([name,current_time])
            #     df = pd.DataFrame(data, columns=['Name', 'Time'])
            #     df.to_excel('name_time.xlsx', index=False)
            #     prev_time = time.time()    
    process_this_frame = not process_this_frame
    # Display the results
    # cv2.namedWindow('Video')
    # cv2.setMouseCallback('Video',mouse_click)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        frame=cv2.putText(frame, name, (left + 14, bottom - 6), font, 0.5, (255, 255, 255), 1)  
        if counter % 2 == 0:
            frame = cv2.putText(frame, "IN", (int(width / 4), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 5)
            frame = cv2.putText(frame, "OUT", (int(3 * width / 4), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 5) 
            frame = cv2.line(frame, (int(width / 2), 0), (int(width / 2), int(height)), (255, 255, 255), 2)
        else:
            frame = cv2.putText(frame,"FACE DETECTED", (int(width / 4), 440), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 5)
            frame = cv2.putText(frame, "CLICK TO CONTINUE", (int(width / 5), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 5)
        cv2.imshow('Video', frame)  
    
        window = tk.Tk()
        window.withdraw()
        window.geometry("6x6")  
        window.mainloop() 
        

    # Display the resulting image
    cv2.imshow('Video', frame)   
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

logging_file.close()