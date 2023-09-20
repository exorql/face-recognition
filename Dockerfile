FROM python:3.8

RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

COPY ./requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./face_recognition.py /app/
COPY ./commons /app/commons

CMD ["bash"]
