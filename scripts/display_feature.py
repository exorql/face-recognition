import sqlite3
import json

# データベースに接続
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# facesテーブルからすべてのデータを取得
cursor.execute("SELECT name, feature FROM faces WHERE id = 1")
rows = cursor.fetchall()

for row in rows:
    name = row[0]
    feature_data = json.loads(row[1])

    print(f"Name: {name}")
    print(f"Feature Data: {feature_data}")
    print("-------------------------------")

conn.close()
