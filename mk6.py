"""
Virtual Attendance System using DeepFace + OpenCV

Features:
- Precomputes embeddings for images in `imagedb/` (one image per person or many images)
- Runs webcam (or process a single image) to detect faces and compute embeddings
- Matches embeddings against the database using cosine distance
- Maintains an attendance CSV with first_seen, last_seen and count

Requirements:
- deepface
- opencv-python
- pandas
- numpy

Usage:
1. Put one or more images of each person inside `imagedb/`.
   Filenames should include the person's name, e.g. `john_doe_1.jpg`, `john_doe_2.png`.
2. Run the script: `python virtual_attendance.py`
3. Press 'q' to quit the webcam loop. Attendance written to `attendance.csv`.

Tweakable:
- DISTANCE_THRESHOLD: lower -> stricter matching (typical 0.4 - 0.6 depending on model)
- MODEL_NAME: model to use for embedding (VGG-Face, Facenet, ArcFace, etc.)
- detector_backend: face detector (retinaface is good)

"""

import os
import time
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from pathlib import Path
from datetime import datetime

# ------------------ CONFIG ------------------
DB_PATH = "imagedb/"            # folder containing reference images
MODEL_NAME = "ArcFace"          # model for embeddings (VGG-Face, Facenet, ArcFace, etc.)
DETECTOR_BACKEND = "retinaface" # detector backend
DISTANCE_THRESHOLD = 0.45        # cosine distance threshold: lower = stricter
ATTENDANCE_CSV = "attendance.csv"
CACHE_EMBEDDINGS = "embeddings_cache.pkl"  # optional cache file
USE_WEBCAM = True                # if False will attempt to process a single image file
SINGLE_IMAGE_PATH = "group.jpg"

# ------------------ HELPERS ------------------

def build_db_embeddings(db_path=DB_PATH, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, cache_file=CACHE_EMBEDDINGS):
    """Iterate over images in db_path, compute embeddings and return a DataFrame with columns: ['identity','path','embedding']"""
    db_path = Path(db_path)
    records = []

    # try to load cache
    if cache_file and os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            print(f"Loaded embeddings cache from {cache_file} ({len(df)} entries)")
            return df
        except Exception:
            print("Failed reading cache, rebuilding embeddings...")

    print("Building embeddings from image database...")
    for p in db_path.rglob("*.jpg"):
        name = p.stem  
        label = "_".join(name.split("_")[:-1]) if len(name.split("_")) > 1 else name
        try:
            emb = DeepFace.represent(str(p), model_name=model_name, detector_backend=detector_backend, enforce_detection=False)
            # DeepFace.represent returns a list for the image when using default api; it may return nested structures
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
                vec = np.array(emb[0]['embedding'], dtype=float)
            elif isinstance(emb, dict) and 'embedding' in emb:
                vec = np.array(emb['embedding'], dtype=float)
            elif isinstance(emb, (list, np.ndarray)):
                vec = np.array(emb, dtype=float)
            else:
                print(f"Unknown embedding format for {p}, skipping")
                continue

            records.append({
                'identity': label,
                'path': str(p),
                'embedding': vec
            })
            print(f"Processed {p} -> {label}")
        except Exception as e:
            print(f"Failed embedding for {p}: {e}")

    df = pd.DataFrame(records)

    if cache_file:
        try:
            df.to_pickle(cache_file)
            print(f"Saved embeddings cache to {cache_file}")
        except Exception as e:
            print("Could not save cache:", e)

    return df


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors (1 - cosine_similarity)"""
    a = a.flatten()
    b = b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - sim


class Attendance:
    def __init__(self, csv_path=ATTENDANCE_CSV):
        self.csv_path = csv_path
        # dict: name -> {first_seen, last_seen, count}
        self.data = {}
        # if exists, load
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                for _, r in df.iterrows():
                    self.data[r['name']] = {
                        'first_seen': r['first_seen'],
                        'last_seen': r['last_seen'],
                        'count': int(r['count'])
                    }
                print(f"Loaded existing attendance ({len(self.data)} records)")
            except Exception:
                print("Failed to load existing attendance, starting fresh.")

    def mark(self, name: str):
        now = datetime.now().isoformat(timespec='seconds')
        if name in self.data:
            self.data[name]['last_seen'] = now
            self.data[name]['count'] += 1
        else:
            self.data[name] = {'first_seen': now, 'last_seen': now, 'count': 1}
        # optionally flush to disk per-mark
        self.save()

    def save(self):
        rows = []
        for name, v in self.data.items():
            rows.append({'name': name, 'first_seen': v['first_seen'], 'last_seen': v['last_seen'], 'count': v['count']})
        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)


# ------------------ MAIN LOOP ------------------

def run_attendance():
    # build / load database embeddings
    db = build_db_embeddings()
    if db.empty:
        print("No reference images found in the database. Exiting.")
        return

    # prepare arrays for quick matching
    db_embs = np.stack(db['embedding'].values)
    db_labels = db['identity'].values

    attendance = Attendance()

    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Starting webcam. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # detect and extract faces using DeepFace.detectFace -> returns aligned face (RGB, 224x224)
            try:
                # DeepFace.detectFace accepts an image (BGR) if passed as numpy array, but requires preprocess later
                faces = []
                # Use DeepFace's find with detector only is slow; we'll try detectFace in a loop by drawing boxes
                # Simpler approach: use DeepFace.detectFace directly on the full frame -> returns largest face. To get multiple faces you'd use a detector library directly.
                face_img = None
                try:
                    # detectFace returns a preprocessed face RGB 224x224 numpy array
                    face_img = DeepFace.detectFace(frame, detector_backend=DETECTOR_BACKEND)
                except Exception:
                    face_img = None

                display_frame = frame.copy()

                if face_img is not None:
                    # get embedding for detected face
                    emb = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)
                    # normalize embedding extraction
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
                        emb_vec = np.array(emb[0]['embedding'], dtype=float)
                    elif isinstance(emb, dict) and 'embedding' in emb:
                        emb_vec = np.array(emb['embedding'], dtype=float)
                    elif isinstance(emb, (list, np.ndarray)):
                        emb_vec = np.array(emb, dtype=float)
                    else:
                        emb_vec = None

                    if emb_vec is not None:
                        # compute distances to all db embeddings
                        dists = np.array([cosine_distance(emb_vec, db_e) for db_e in db_embs])
                        best_idx = np.argmin(dists)
                        best_dist = float(dists[best_idx])
                        best_name = db_labels[best_idx]

                        if best_dist < DISTANCE_THRESHOLD:
                            label = f"{best_name} ({best_dist:.2f})"
                            attendance.mark(best_name)
                        else:
                            label = f"Unknown ({best_dist:.2f})"

                        # put label on top-left corner
                        cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    else:
                        cv2.putText(display_frame, "Face found - no embedding", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                else:
                    cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            except Exception as e:
                print("Error during face detection/recognition:", e)
                display_frame = frame

            cv2.imshow("Virtual Attendance", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # process single image
        frame = cv2.imread(SINGLE_IMAGE_PATH)
        if frame is None:
            print("Could not open image", SINGLE_IMAGE_PATH)
            return
        face_img = DeepFace.detectFace(frame, detector_backend=DETECTOR_BACKEND)
        if face_img is not None:
            emb = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
                emb_vec = np.array(emb[0]['embedding'], dtype=float)
            elif isinstance(emb, dict) and 'embedding' in emb:
                emb_vec = np.array(emb['embedding'], dtype=float)
            elif isinstance(emb, (list, np.ndarray)):
                emb_vec = np.array(emb, dtype=float)
            else:
                emb_vec = None

            if emb_vec is not None:
                dists = np.array([cosine_distance(emb_vec, db_e) for db_e in db_embs])
                best_idx = np.argmin(dists)
                best_dist = float(dists[best_idx])
                best_name = db_labels[best_idx]

                if best_dist < DISTANCE_THRESHOLD:
                    print("Matched:", best_name, best_dist)
                    attendance.mark(best_name)
                else:
                    print("Unknown, best dist:", best_dist)

        attendance.save()


if __name__ == '__main__':
    run_attendance()
