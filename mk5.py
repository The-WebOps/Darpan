from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd

img_path = "group.jpg"
path="imagedb/"

# Detect and crop each face

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    detections = DeepFace.find(img_path=frame, db_path="imagedb/",detector_backend='retinaface', enforce_detection=False)
    for i in range(len(detections)):
        if len(detections) > 0:
            df = detections[i]
            identities = df["identity"].tolist()
            print(identities)
        else:
            print("No faces found in the database.")

    



























# df=pd.DataFrame(detections)
# df = pd.DataFrame(detections, columns=["identity", "confidence"])

# print(df)


# for face in detections:
#     x, y, w, h = face['facial_area'].values()
#     face_img = face['face']
#     # Recognize face
#     result = DeepFace.find(img_path=face_img, db_path=path, enforce_detection=True)
#     # Draw a rectangle and label
#     cv2.rectangle(img_path, (x, y), (x+w, y+h), (0,255,0), 2)
#     if not result.empty:
#         name = result.iloc[0]["identity"].split("/")[-2]  # folder name as label
#         cv2.putText(img_path, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# cv2.imshow("Face Recognition", img_path)
# cv2.waitKey(0)
# cv2.destroyAllWindows()