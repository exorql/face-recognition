import cv2
import time
import numpy as np
import sqlite3
from deepface import DeepFace
import json
from sklearn.metrics.pairwise import cosine_similarity
from deepface.commons import functions

cap = cv2.VideoCapture(0)

# Initialize the database
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    name TEXT,
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
recognition_interval = 5
target_size = (224, 224)

# Variable for determining the same person 5 times in a row
last_recognized_name = ""
continuous_recognition_count = 0
recognition_threshold = 1
current_display_name= ""
freeze_start_time = None
freeze_duration = 5
freezed_frame = None

non_matched_count = 0
last_face_position = None
last_name_position = None

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    face_embedding = None
    frame_count += 1

    if frame_count % recognition_interval == 0:

      face_objs = DeepFace.extract_faces(img_path=frame, target_size=target_size, enforce_detection=False)
      faces = [(obj["facial_area"]["x"], obj["facial_area"]["y"], obj["facial_area"]["w"], obj["facial_area"]["h"]) for obj in face_objs if obj.get('confidence', 0) > 5]
      if freeze_start_time and time.time() - freeze_start_time > freeze_duration:
          freeze_start_time = None
          last_recognized_name = ""

      if faces:
          areas = [w*h for (x, y, w, h) in faces]
          max_area_index = areas.index(max(areas))
          x, y, w, h = faces[max_area_index]
          face = frame[y:y+h, x:x+w]
          face_detected = True
          face_included_frames += 1

          if not freeze_start_time:
              face_embedding = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)
              if face_detected and face_embedding:
                  recognized_names = []
                  for row in cursor.execute("SELECT name, feature FROM faces"):
                      saved_name = row[0]
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
                      similarity = cosine_similarity(face_embedding_array.reshape(1, -1), saved_feature.reshape(1, -1))
                      print(f"{similarity[0][0]}")
                      if similarity[0][0] > 0.6:
                          recognized_names.append(saved_name)

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
                      if non_matched_count >= 3 and face_detected:
                          if face_embedding:
                              customer_name = input("Enter Customer Name: ")
                              binary_feature = json.dumps(face_embedding)
                              cursor.execute("INSERT INTO faces (name, feature) VALUES (?, ?)", (customer_name, binary_feature))
                              conn.commit()
                              print(f"新しい顧客 '{customer_name}' のデータを保存しました。")
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
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
