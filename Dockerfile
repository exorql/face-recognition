# ベースイメージの選択
FROM python:3.8

# ディレクトリの作成と移動
RUN mkdir /app
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libopencv-dev \
    python3-opencv \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール
COPY ./requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# スクリプトのコピー
COPY ./face_recognition.py /app/
COPY ./commons /app/commons

CMD ["python3", "face_recognition.py"]
