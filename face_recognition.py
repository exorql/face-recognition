import cv2
import numpy as np
import sqlite3
from deepface import DeepFace
import json
from sklearn.metrics.pairwise import cosine_similarity

cap = cv2.VideoCapture(0)
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
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
DeepFace.build_model(model_name="VGG-Face")
face_saved = False
first_time_customer = False
customer_name = ""
recognized_customers = []

def save_new_customer(face_embedding, customer_name):
    binary_feature = json.dumps(face_embedding)
    cursor.execute("INSERT INTO faces (name, feature) VALUES (?, ?)", (customer_name, binary_feature))
    conn.commit()
    print(f"新しい顧客 '{customer_name}' のデータを保存しました。")
    recognized_customers.append(customer_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and confidence < 1.0:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            # 顔の周りに四角形を描画
            cv2.rectangle(frame, (x, y), (x + int(w * 0.3), y + int(h * 0.6)), (0, 255, 0), 1)
            face = frame[y:y + h, x:x + w]
            face_embedding = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)
            if first_time_customer:
                cv2.putText(frame, "Enter Customer Name:", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                if customer_name == "":
                    customer_name = input("Enter Customer Name: ")
                else:
                    save_new_customer(face_embedding, customer_name)
                    first_time_customer = False
            else:
                recognized_names = []
                for row in cursor.execute("SELECT name, feature FROM faces"):
                    saved_name = row[0]
                    saved_feature_list = json.loads(row[1])  # BLOBデータをJSON形式からPythonリストに変換
                    saved_feature_dict = saved_feature_list[0]
                    saved_feature = np.array(saved_feature_dict['embedding'], dtype=np.float32)
                    face_embedding_dict = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)
                    face_embedding = np.array(face_embedding_dict[0]['embedding'], dtype=np.float32)
                    if face_embedding.shape != saved_feature.shape:
                        face_embedding = face_embedding.reshape(saved_feature.shape)
                    similarity = cosine_similarity(face_embedding.reshape(1, -1), saved_feature.reshape(1, -1))

                    # 閾値の調整
                    if similarity[0][0] > 0.8:
                        print(similarity[0][0])
                        recognized_names.append(saved_name)
                if recognized_names:
                    print(f"Welcome {recognized_names[0]}")
                    cv2.putText(frame, f"Match {recognized_names[0]}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{similarity[0][0]}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                else:
                    print("Does not match")
                    cv2.putText(frame, f"Does not match", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
