import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import time
import subprocess
import sys

# 링 버퍼 초기화
result_buffer = []

USE_WEBCAM = True  # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# 제한 시간 설정 (초 단위)
time_limit = 3

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')  # Video file source

start_time = time.time()


while cap.isOpened():
    # 현재 시간 계산
    current_time = time.time()

    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else :
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

        # Display satisfaction in cmd
        print(f"표정: {emotion_text}")

        # 표정과 만족도 결과를 링 버퍼에 추가합니다
        result_buffer.append((emotion_text))

        # 링 버퍼 크기 유지
        if len(result_buffer) > 5:
            result_buffer.pop(0)

# 제한 시간을 초과하면 종료
    if current_time - start_time > time_limit:
        print("\n캡처된 결과:")
        for emotion in result_buffer:
            print(f"표정: {emotion}")

        # 결과를 파일로 저장
        with open('emotion_check.txt', 'w', encoding="utf-8") as f:
            for result in result_buffer:
                emotion_text = result
                f.write(f"Emotion: {emotion_text}\n")
        # 웹캠이 가려지지 않고 5초가 지난 경우 AI.py 파일 실행
        subprocess.Popen(['python', 'AI_First.py'])
        break

    if len(faces) == 0:
        print("\n캡처된 결과:")
        for emotion in result_buffer:
            print(f"표정: {emotion}")

        # 결과를 파일로 저장
        with open('emotion_check.txt', 'w', encoding="utf-8") as f:
            for result in result_buffer:
                emotion_text = result
                f.write(f"Emotion: {emotion_text}\n")
        subprocess.Popen(['python', 'STT.py'])

        break






cap.release()
cv2.destroyAllWindows()