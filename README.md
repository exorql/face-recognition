# face-recognition

Face recognition system using various deep learning models and SQLite database for storing recognized faces.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the face recognition system, use the following command:

```bash
python3 face_recognition.py [OPTIONS]
```

### Options:

- `-d`, `--detector_backend`: Detector backend to use. Choices are ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet']. Default is 'opencv'.

- `-m`, `--model_name`: Face recognition model to use. Choices are ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib']. Default is 'VGG-Face'.

- `-dm`, `--distance_metric`: Distance metric for face similarity. Choices are ['cosine', 'euclidean', 'euclidean_l2']. Default is 'euclidean'.

- `-s`, `--source`: Camera source index or video file path. Default is '0' (webcam).

### Examples:

To use the default settings:

```bash
python3 face_recognition.py
```

To use the 'mediapipe' detector backend and 'Facenet' model:

```bash
python3 face_recognition.py -d 'retinaface'
```

## Docker Setup

### Building the Docker Image:

```bash
docker build -t face_recognition_image .
```

## Running the Docker Container:

### For Ubuntu
Before running the container, ensure you allow Docker to access the X window:

```bash
sudo xhost +si:localuser:root
```

```bash
docker run -it --rm --name face_recognition_image --network host -e DISPLAY=$DISPLAY --device /dev/video0:/dev/video0 face_recognition_image
```
