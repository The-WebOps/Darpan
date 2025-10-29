import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(1)
db_path = "imagedb/"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    try:
        detections = DeepFace.extract_faces(frame, detector_backend='')
        for face in detections:
            x, y, w, h = face['facial_area'].values()
            face_img = face['face']
            # Recognize face
            result = DeepFace.find(img_path=face_img, db_path=db_path, enforce_detection=True)
            # Draw a rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            if not result.empty:
                name = result.iloc[0]["identity"].split("/")[-2]  # folder name as label
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    except:
        pass

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()