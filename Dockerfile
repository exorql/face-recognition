FROM nvcr.io/nvidia/l4t-base:r32.7.1

WORKDIR /app

ENV TZ Asia/Tokyo
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    software-properties-common \
    libopencv-dev \
    python3-opencv

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

RUN apt-get install -y python3.8 python3.8-dev python3.8-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN apt-get install -y wget && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

RUN pip3 install numpy deepface

COPY ./face_recognition.py /app/

CMD ["python3", "face_recognition.py"]
