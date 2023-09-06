import cv2
import numpy as np
import sqlite3
from deepface import DeepFace
import json
from sklearn.metrics.pairwise import cosine_similarity

# カメラの初期化
cap = cv2.VideoCapture(0)

# カスケード分類器の初期化
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# データベースの初期化
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

# 変数の初期化
face_detected = False
face_included_frames = 0
frame_threshold = 10
customer_name = ""
frame_count = 0
recognition_interval = 5  # 5フレームごとに顔認識を行う

# 5回連続で同一人物として判定するための変数
last_recognized_name = ""
continuous_recognition_count = 0
recognition_threshold = 5


# 名前が表示されるフレーム数を設定
display_name_duration = 50  # 例: 50フレーム
display_name_counter = 0
display_name = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        if frame_count % recognition_interval == 0:
            face_embedding = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)

            if face_detected:
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
                    if similarity[0][0] > 0.6:
                        recognized_names.append(saved_name)

                if recognized_names:
                    if recognized_names[0] == last_recognized_name:
                        continuous_recognition_count += 1
                    else:
                        continuous_recognition_count = 1
                        last_recognized_name = recognized_names[0]

                    if continuous_recognition_count >= recognition_threshold:
                        cv2.putText(frame, f"Welcome {last_recognized_name}", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                        print(f"Welcome {last_recognized_name}")
                        continuous_recognition_count = 0  # カウントをリセット
                else:
                    continuous_recognition_count = 0  # 顔が認識されなかった場合、カウントをリセット
                    last_recognized_name = ""
                    if customer_name == "":
                        customer_name = input("Enter Customer Name: ")
                        binary_feature = json.dumps({'embedding': face_embedding[0]['embedding']})
                        cursor.execute("INSERT INTO faces (name, feature) VALUES (?, ?)", (customer_name, binary_feature))
                        conn.commit()
                        print(f"新しい顧客 '{customer_name}' のデータを保存しました。")
                        customer_name = ""
                    face_detected = False
                    face_included_frames = 0
            else:
                face_included_frames += 1
                if face_included_frames == frame_threshold:
                    face_detected = True
    # 名前の表示制御をforループの外に移動
    if continuous_recognition_count >= recognition_threshold:
        display_name = f"Welcome {last_recognized_name}"
        if display_name_counter == 0:  # この行を追加
            display_name_counter = display_name_duration  # 名前の表示カウンタを設定
        continuous_recognition_count = 0  # カウントをリセット

    if display_name_counter > 0:
        # 名前の表示カウンタが0より大きい場合、名前を表示し続ける
        cv2.putText(frame, display_name, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        display_name_counter -= 1
    cv2.imshow('Camera', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
