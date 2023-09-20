FROM python:3.8-slim-buster

RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./face_recognition.py /app/
COPY ./commons /app/commons

CMD ["bash"]
