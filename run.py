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
import os
import datetime
from tensorflow.keras.models import load_model

mask_detector_model_path = "/Users/watanabeharuto/sample/face-recognition/Face-Mask-Detection/mask_detector.model"
mask_detector = load_model(mask_detector_model_path)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

parser = argparse.ArgumentParser(description='Face recognition system.')
parser.add_argument('-d', '--detector_backend', type=str, default='yolov8', choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet'], help='Detector backend to use.')
parser.add_argument('-m', '--model_name', type=str, default="Facenet512", choices=['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'], help="Face recognition model.")
parser.add_argument('-dm', '--distance_metric', type=str, default="cosine", choices=['cosine', 'euclidean', 'euclidean_l2'], help="Distance metric for face similarity.")
parser.add_argument('-s', '--source', type=str, default='0', help='Camera source index or video file path')

args = parser.parse_args()

detector_backend = args.detector_backend
model_name = args.model_name
distance_metric = args.distance_metric
source = args.source

cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
fps = int(cap.get(cv2.CAP_PROP_FPS)) if not source.isdigit() else None
# if os.path.exists('faces.db'):
#     os.remove('faces.db')
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                  (id INTEGER PRIMARY KEY,
                   number INTEGER,
                   feature BLOB,
                   last_matched TIMESTAMP)''')
conn.commit()

image_save_dir = '/Users/watanabeharuto/sample/face-recognition/yoloface'
os.makedirs(image_save_dir, exist_ok=True)

def detect_mask(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, 224, 224, 3)
    # print(mask_detector.predict(face_img)[0][0])
    # log_to_file("mask_detector: " + str(mask_detector.predict(face_img)[0][0]))
    return mask_detector.predict(face_img)[0][0] > 0.98

def calculate_similarity(face_embedding, saved_feature, distance_metric):
    face_embedding_array = np.array(face_embedding[0]['embedding'], dtype=np.float32) if isinstance(face_embedding, list) else np.array(face_embedding['embedding'], dtype=np.float32)
    face_embedding_array = face_embedding_array.reshape(saved_feature.shape) if face_embedding_array.shape != saved_feature.shape else face_embedding_array
    if distance_metric == "cosine":
        return 1 - dst.findCosineDistance(face_embedding_array, saved_feature)
    elif distance_metric == "euclidean":
        return -dst.findEuclideanDistance(face_embedding_array, saved_feature)
    elif distance_metric == "euclidean_l2":
        return -dst.findEuclideanDistance(dst.l2_normalize(face_embedding_array), dst.l2_normalize(saved_feature))
    else:
        raise ValueError(f"Invalid distance_metric: {distance_metric}")

def log_to_file(message):
    with open("face_recognition_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def is_face_forward(face_landmarks):
    HORIZONTAL_THRESHOLD = 0.20
    VERTICAL_LEFT_THRESHOLD = 0.09
    CHIN_NOSE_THRESHOLD = 0.18
    EYE_DISTANCE_THRESHOLD = 0.5
    SIDE_FACE_THRESHOLD = 0.70  # 横を向いている場合のしきい値
    VERTICAL_RIGHT_THRESHOLD_UPPER = 0.35  # 下を向いていると判断するしきい値の上限
    VERTICAL_RIGHT_THRESHOLD_LOWER = 0.05

    horizontal_difference = abs(face_landmarks.landmark[33].x - face_landmarks.landmark[263].x)
    vertical_difference_left = abs(face_landmarks.landmark[4].y - ((face_landmarks.landmark[234].y + face_landmarks.landmark[124].y) / 2))
    vertical_difference_right = abs(face_landmarks.landmark[4].y - ((face_landmarks.landmark[454].y + face_landmarks.landmark[344].y) / 2))
    chin_nose_difference = face_landmarks.landmark[152].y - face_landmarks.landmark[4].y
    left_eye_center = face_landmarks.landmark[33]
    right_eye_center = face_landmarks.landmark[263]
    eye_distance = ((left_eye_center.x - right_eye_center.x) ** 2 + (left_eye_center.y - right_eye_center.y) ** 2) ** 0.5

    return (
        abs(horizontal_difference) <= SIDE_FACE_THRESHOLD and
        abs(horizontal_difference) >= HORIZONTAL_THRESHOLD and
        abs(vertical_difference_left) >= VERTICAL_LEFT_THRESHOLD and
        VERTICAL_RIGHT_THRESHOLD_LOWER <= abs(vertical_difference_right) <= VERTICAL_RIGHT_THRESHOLD_UPPER and
        abs(chin_nose_difference) >= CHIN_NOSE_THRESHOLD and
        abs(eye_distance) >= EYE_DISTANCE_THRESHOLD
    )


target_size = functions.find_target_size(model_name=model_name)
frame_count = 0
recognition_interval = 5
last_recognized_name = ""
last_face_position = None
continuous_recognition_count = 0
freeze_start_time = None
freeze_duration = 3
recognition_threshold = 1
new_sequence = 0
non_matched_count = 0
face_included_frames = 0
area_threshold = 20000
NEW_FACE_SIMILARITY_THRESHOLD = 0.55

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_count += 1
    if frame_count % recognition_interval == 0:
      face_objs = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, target_size=target_size, enforce_detection=False, align=True)
      # confidence check
      faces = [(obj["facial_area"]["x"], obj["facial_area"]["y"], obj["facial_area"]["w"], obj["facial_area"]["h"]) for obj in face_objs if obj.get('confidence', 0) > 0.5]

      if faces:
          areas = [w*h for (x, y, w, h) in faces]
          max_area_index = areas.index(max(areas))
          x, y, w, h = faces[max_area_index]

          max_area = w*h
          if max_area < area_threshold:
              print("Detected face is too far, skipping recognition.")
              continue
          face = frame[y:y+h, x:x+w]
          face_detected = True
          mask_detected = False
          skip_save = True

          mask_detected = detect_mask(face)
          if mask_detected:
              last_recognized_name = ""
              print("Mask detected, skipping face recognition.")
              continue

          face_landmarks = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
          if face_landmarks.multi_face_landmarks:
            if is_face_forward(face_landmarks.multi_face_landmarks[0]):
                print("Face is forward, proceeding with recognition.")
                log_to_file(f"Face is forward, proceeding with recognition.")
                skip_save = False
            else:
                # print("Face is not forward, skipping recognition.")
                log_to_file(f"Face is not forward, skipping recognition.")
                skip_save = True
          else:
              skip_save = True
              print("No landmarks detected, proceeding with recognition.")

          if not freeze_start_time:
              face_embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)
              recognized_names = []
              distances = []
              saved_names = []
              saved_features = []
              max_similarity = 0
              rows = list(cursor.execute("SELECT number, feature FROM faces"))

              # DBに顔データがない場合、現在の顔データを追加
              if not rows and not skip_save:
                  binary_feature = json.dumps(face_embedding)
                  current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                  cursor.execute("INSERT INTO faces (number, feature, last_matched) VALUES (?, ?, ?)", (1, binary_feature, current_time))
                  conn.commit()
                  cropped_face = frame[y:y+h, x:x+w]
                  image_filename = f"{new_sequence}.png"
                  image_filepath = os.path.join(image_save_dir, image_filename)
                  cv2.imwrite(image_filepath, cropped_face)
                  log_to_file(f"新しい顔の画像を保存しました。ファイルパス: {image_filepath}")
                  log_to_file(f"新しい顔のデータを保存しました。連番: {new_sequence}")
              for row in rows:
                  saved_name = row[0]
                  saved_names.append(saved_name)
                  saved_data = json.loads(row[1])
                  if isinstance(saved_data, list):
                      saved_feature = np.array(saved_data[0]['embedding'], dtype=np.float32)
                  else:
                      saved_feature = np.array(saved_data['embedding'], dtype=np.float32)
                  saved_features.append(saved_feature)

                  similarity = calculate_similarity(face_embedding, saved_feature, distance_metric)
                  distances.append(similarity)
                  print(f"distances: {similarity}")

              if distances:
                  max_similarity = max(distances)
                  best_face_index = np.argmax(distances)
                  if best_face_index < len(faces):
                      best_face_area = faces[best_face_index]
                  threshold_similarity = dst.findThreshold(model_name, distance_metric)
                  log_to_file(f"max_similarity {max_similarity}")
                  if max_similarity > threshold_similarity:
                      cropped_face = frame[y:y+h, x:x+w]
                      image_filename = f"matched_{last_recognized_name}_{frame_count}.png"
                      image_filepath = os.path.join(image_save_dir, image_filename)
                      print(f"skip_save value: {skip_save}")

                      log_to_file(f"Match {image_filepath}")
                      log_to_file(f"Match {new_sequence}")
                      cv2.imwrite(image_filepath, frame)

                      current_time = datetime.datetime.now()
                      cursor.execute("UPDATE faces SET last_matched = ? WHERE number = ?", (current_time, best_face_index))
                      conn.commit()
                      face_detected = False
                      non_matched_count = 0

              if face_detected and face_embedding and not skip_save:
                  # max_similarity = 0
                  # if distances:
                  #     max_similarity2 = max(distances)
                  log_to_file(f"max_similarity2: {max_similarity}")
                  print(max_similarity)
                  if max_similarity < NEW_FACE_SIMILARITY_THRESHOLD:
                      print(f"non_matched_count {non_matched_count}")
                      if non_matched_count > 1:
                          cursor.execute("SELECT MAX(number) FROM faces")
                          max_sequence = cursor.fetchone()[0]
                          if max_sequence is None:
                              max_sequence = 0

                          new_sequence = max_sequence + 1
                          binary_feature = json.dumps(face_embedding)
                          current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                          cursor.execute("INSERT INTO faces (number, feature, last_matched) VALUES (?, ?, ?)", (new_sequence, binary_feature, current_time))
                          conn.commit()
                          cropped_face = frame[y:y+h, x:x+w]
                          image_filename = f"{new_sequence}.png"
                          image_filepath = os.path.join(image_save_dir, image_filename)
                          cv2.imwrite(image_filepath, cropped_face)
                          print(f"skip_save value: {skip_save}")

                          log_to_file(f"新しい顔の画像を保存しました。ファイルパス: {image_filepath}")
                          log_to_file(f"新しい顔のデータを保存しました。連番: {new_sequence}")


                          non_matched_count = 0
                      else:
                          non_matched_count += 1

                  face_detected = False
          expand_margin = 40
          x_start = max(x - expand_margin, 0)
          y_start = max(y - expand_margin, 0)
          x_end = min(x + w + expand_margin, frame.shape[1])
          y_end = min(y + h + expand_margin, frame.shape[0])
          last_face_position = (x_start, y_start, x_end, y_end)

          # font_scale = 1.6
          # if freeze_start_time and time.time() - freeze_start_time <= freeze_duration:
          #     cv2.putText(frame, f"Match {last_recognized_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
          #     last_name_position = (x, y)

              # 画像を保存する
              # image_filename = f"matched_{last_recognized_name}_{frame_count}.png"  # 画像のファイル名（マッチした名前とフレームカウントを含む）
              # image_filepath = os.path.join(image_save_dir, image_filename)  # 保存するパス
              # cv2.imwrite(image_filepath, frame)  # 画像を保存
              # log_to_file(f"マッチした顔の画像を保存しました。ファイルパス: {image_filepath}")  # ログに保存の情報を追加

      else:
          last_face_position = None
          face_detected = False
          face_included_frames = 0
      # if freeze_start_time and last_matched_name and last_name_position:
      #   x, y = last_name_position
      #   cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
      #   cv2.putText(frame, f"Match {last_matched_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    # if last_face_position:
    #     x_start, y_start, x_end, y_end = last_face_position
    #     cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
