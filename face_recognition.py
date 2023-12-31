import cv2
import time
import numpy as np
import sqlite3
from deepface import DeepFace
import json
from deepface.commons import functions
from commons import distance as dst
import argparse
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser(description='Face recognition system using Jetson Nano.')

# Arguments
parser.add_argument('-d', '--detector_backend', type=str, default='mediapipe', choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet'], help='Detector backend to use.')
parser.add_argument('-m', '--model_name', type=str, default="Facenet", choices=['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'], help="Face recognition model.")
parser.add_argument('-dm', '--distance_metric', type=str, default="cosine", choices=['cosine', 'euclidean', 'euclidean_l2'], help="Distance metric for face similarity.")
parser.add_argument('-s', '--source', type=str, default='0', help='Camera source index or video file path')

args = parser.parse_args()

# Variables from arguments
detector_backend = args.detector_backend
model_name = args.model_name
distance_metric = args.distance_metric
source = args.source

# Open video source
if source.isdigit():
    cap = cv2.VideoCapture(int(source))
else:
    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the database
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    number INTEGER,
    feature BLOB
)
''')
conn.commit()

# Initializing variables
face_detected = False
face_included_frames = 0
frame_threshold = 5
customer_name = ""
frame_count = 0
recognition_interval = 1  # Changed to 1 for testing
target_size = functions.find_target_size(model_name=model_name)
last_recognized_name = ""
continuous_recognition_count = 0
recognition_threshold = 1
current_display_name = ""
freeze_start_time = None
freeze_duration = 5
freezed_frame = None
non_matched_count = 0
last_face_position = None
last_name_position = None
new_sequence = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
distances = []

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    # 顔のランドマーク検出
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    threshold = 0.15
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[4]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]

            nose_to_left_eye = np.array([nose_tip.x - left_eye_outer.x, nose_tip.y - left_eye_outer.y])
            nose_to_right_eye = np.array([nose_tip.x - right_eye_outer.x, nose_tip.y - right_eye_outer.y])

            if abs(nose_to_left_eye[0]) > threshold or abs(nose_to_right_eye[0]) > threshold:
                continue

    face_embedding = None
    frame_count += 1

    if frame_count % recognition_interval == 0:
      face_objs = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, target_size=target_size, enforce_detection=False)
      # confidence check
      faces = [(obj["facial_area"]["x"], obj["facial_area"]["y"], obj["facial_area"]["w"], obj["facial_area"]["h"]) for obj in face_objs if obj.get('confidence', 0) > 0.9]
      if freeze_start_time and time.time() - freeze_start_time > freeze_duration:
          freeze_start_time = None
          last_recognized_name = ""

      if faces:
          areas = [w*h for (x, y, w, h) in faces]
          max_area_index = areas.index(max(areas))
          x, y, w, h = faces[max_area_index]
          face = frame[y:y+h, x:x+w]
          # emotions分析
          # results = DeepFace.analyze(img_path=face, actions=['emotion'], enforce_detection=False)
          # print(results)
          # emotions = results[0]["emotion"]
          # if emotions["happy"] > 50 or emotions["surprise"] > 50:
          #     continue
          face_detected = True
          face_included_frames += 1

          if not freeze_start_time:
              # 指定されたモデル（model_name）を使用して、入力された顔画像（face）の特徴ベクトル（embedding）を取得
              face_embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)

              if face_detected and face_embedding:
                  recognized_names = []
                  distances = []
                  saved_names = []
                  for row in cursor.execute("SELECT number, feature FROM faces"):
                      saved_name = row[0]
                      saved_names.append(saved_name)
                      saved_data = json.loads(row[1])
                      if isinstance(saved_data, list):
                          saved_feature = np.array(saved_data[0]['embedding'], dtype=np.float32)
                      else:
                          saved_feature = np.array(saved_data['embedding'], dtype=np.float32)

                      if isinstance(face_embedding, list):
                          face_embedding_array = np.array(face_embedding[0]['embedding'], dtype=np.float32)
                      else:
                          face_embedding_array = np.array(face_embedding['embedding'], dtype=np.float32)

                      if face_embedding_array.shape != saved_feature.shape:
                          face_embedding_array = face_embedding_array.reshape(saved_feature.shape)

                      if distance_metric == "cosine":
                          similarity = 1 - dst.findCosineDistance(face_embedding_array, saved_feature)
                      elif distance_metric == "euclidean":
                          similarity = -dst.findEuclideanDistance(face_embedding_array, saved_feature)
                      elif distance_metric == "euclidean_l2":
                          similarity = -dst.findEuclideanDistance(dst.l2_normalize(face_embedding_array), dst.l2_normalize(saved_feature))

                      distances.append(similarity)

                  if distances:
                      max_similarity = max(distances)
                      best_face_index = np.argmax(distances)
                      if best_face_index < len(faces):
                          best_face_area = faces[best_face_index]
                      threshold_similarity = dst.findThreshold(model_name, distance_metric)
                      print(f"最も類似性の高い距離: {max_similarity}")
                      best_similarity = max_similarity
                      if best_similarity > threshold_similarity:
                          recognized_name = saved_names[best_face_index]
                          recognized_names.append(recognized_name)

                  if recognized_names:
                      if recognized_names[0] == last_recognized_name:
                          continuous_recognition_count += 1
                      else:
                          continuous_recognition_count = 1
                          last_recognized_name = recognized_names[0]

                      if continuous_recognition_count >= recognition_threshold:
                          print(f"Match {last_recognized_name}")
                          continuous_recognition_count = 0
                          freeze_start_time = time.time()
                      face_detected = False
                      face_included_frames = 0
                      non_matched_count = 0
                  else:
                      continuous_recognition_count = 0
                      last_recognized_name = ""

                      non_matched_count += 1
                      if non_matched_count >= 5 and face_detected:
                          if face_embedding:
                              cursor.execute("SELECT MAX(number) FROM faces")
                              max_sequence = cursor.fetchone()[0]
                              if max_sequence is None:
                                  max_sequence = 0

                              new_sequence = max_sequence + 1

                              binary_feature = json.dumps(face_embedding)
                              cursor.execute("INSERT INTO faces (number, feature) VALUES (?, ?)", (new_sequence, binary_feature))
                              conn.commit()
                              print(f"新しい顔のデータを保存しました。連番: {new_sequence}")
                          face_detected = False
                          face_included_frames = 0
                          non_matched_count = 0

          expand_margin = 40
          x_start = max(x - expand_margin, 0)
          y_start = max(y - expand_margin, 0)
          x_end = min(x + w + expand_margin, frame.shape[1])
          y_end = min(y + h + expand_margin, frame.shape[0])
          last_face_position = (x_start, y_start, x_end, y_end)

          font_scale = 1.6
          if freeze_start_time:
              cv2.putText(frame, f"Match {last_recognized_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
              last_name_position = (x, y)
      else:
          last_face_position = None
          last_name_position = None
          face_detected = False
          face_included_frames = 0
    if last_name_position and freeze_start_time:
        x, y = last_name_position
        cv2.putText(frame, f"Match {last_recognized_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    if last_face_position:
        x_start, y_start, x_end, y_end = last_face_position
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    if source.isdigit():  # If the source is a webcam
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:  # If the source is a video file
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
