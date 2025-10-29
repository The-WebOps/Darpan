import cv2
import numpy as np

# Load models
detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Load and prepare image
image = cv2.imread("image.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             1.0, (300, 300), (104.0, 177.0, 123.0))

# Detect face
detector.setInput(blob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Extract face ROI
        face = image[startY:endY, startX:endX]
        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96),
                                          (0, 0, 0), swapRB=True, crop=False)

        # Get 128D embedding
        embedder.setInput(face_blob)
        vec = embedder.forward()  # shape: (1, 128)

        print("Embedding vector:", vec.shape)