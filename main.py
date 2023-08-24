import os, sys
import math 
import cv2 #display video capture
import numpy as np 
import face_recognition as fr #need for encodings(), locations()

def face_accuracy( face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    if face_distance > face_match_threshold: 
        return str(round(linear_val * 100, 2)) + '%'
    else: 
        value = (linear_val + ((1.0-linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition: 
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
    
    #checks the faces folder to return the items in the folder
    def encode_faces(self):
        for img in os.listdir('faces'):
            face_img = fr.load_image_file(f'faces/{img}')
            face_encodings = fr.face_encodings(face_img)[0] 

            self.known_face_encodings.append(face_encodings)
            self.known_face_names.append(img)
        print(face_img)
        print(self.known_face_names)
    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Video source not found ❌')
        while True: 
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                #find all faces in the current frame
                self.face_locations= fr.face_locations(rgb_small_frame)
                self.face_encodings= fr.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                
                #checks if the face is known
                for face_encoding in self.face_encodings:
                    matches = fr.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    accuracy = 'Unknown'

                    face_dist = fr.face_distance(self.known_face_encodings, face_encoding)
                    best_match_idx = np.argmin(face_dist)

                    if matches[best_match_idx]:
                        name = self.known_face_names[best_match_idx]
                        accuracy = face_accuracy(face_dist[best_match_idx])

                    self.face_names.append(f'{name}({accuracy})')

            self.process_current_frame = not self.process_current_frame #refreshes

            #display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *=4
                right *=4
                bottom *=4
                left *=4
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,225), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,225), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xff== ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__': 
    try:
        face_detection = FaceRecognition()
        face_detection.run_recognition()
    except Exception as e:
        print('Error:', e)