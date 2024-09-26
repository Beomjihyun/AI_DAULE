import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
import time
from emotions import face_cascade
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import threading

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

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')  # Video file source

# 웹캠 스레드 클래스
class WebcamThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.paused = False
        self.pause_condition = threading.Condition()

    def run(self):
        global result_buffer

        while self.running:
            with self.pause_condition:
                while self.paused:
                    self.pause_condition.wait()

            ret, bgr_image = self.cap.read()

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
                    satisfaction = 'bad'
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    satisfaction = 'bad'
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    satisfaction = 'good'
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    satisfaction = 'not bad'
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    satisfaction = 'not bad'

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)
                # Display satisfaction in cmd
                print(f"Emotion: {emotion_text}")
                print(f"Satisfaction: {satisfaction}")

                # 표정과 만족도 결과를 링 버퍼에 추가합니다
                result_buffer.append((emotion_text, satisfaction))

                # 링 버퍼 크기 유지
                if len(result_buffer) > 5:
                    result_buffer.pop(0)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('window_frame', bgr_image)



        self.cap.release()
        cv2.destroyAllWindows()

    def pause(self):
        self.paused = True

    def resume(self):
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify()

    def stop(self):
        self.running = False



import threading
import speech_recognition as sr
from playsound import playsound

class STTThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.r = sr.Recognizer()
        self.is_listening = False
        self.result_text = None

    def run(self):
        with sr.Microphone() as source:
            print("음성 인식 대기 중...")

            while True:
                if self.is_listening:
                    audio = self.r.listen(source)

                    try:
                        text = self.r.recognize_google(audio, language='ko-KR')

                        if '에이' in text:
                            playsound('ding.wav')
                            print("다음 음성을 기다리는 중...")
                            audio = self.r.listen(source, phrase_time_limit=5)
                            text = self.r.recognize_google(audio, language='ko-KR')

                        self.result_text = text.replace("에이", "")

                    except sr.UnknownValueError:
                        print("음성을 인식할 수 없습니다.")
                    except sr.RequestError:
                        print("Google 음성 인식 서비스에 접근할 수 없습니다.")

    def start_listening(self):
        self.is_listening = True

    def stop_listening(self):
        self.is_listening = False

    def get_result(self):
        return self.result_text

# STT 스레드 생성
stt_thread = STTThread()

# 웹캠 스레드 생성
webcam_thread = WebcamThread()

# 스레드 시작
webcam_thread.start()


stt_thread.start()

# # STT 스레드 시작
# stt_thread.start()

# STT 스레드로 음성 입력 시작
stt_thread.start_listening()

# 일시정지 상태에서 다시 키 입력 대기
if stt_thread.start_listening == True:
        webcam_thread.pause()
        # 캡처된 결과 출력
        print("캡처된 결과:")
        for emotion, satisfaction in result_buffer:
            print(f"표정: {emotion}, 만족도: {satisfaction}")

        # 결과를 파일로 저장
        with open('captured_results.txt', 'w') as f:
            for result in result_buffer:
                emotion_text, satisfaction = result
                f.write(f"Emotion: {emotion_text}\n")
                f.write(f"Satisfaction: {satisfaction}\n")



# 일정 시간 동안 음성 입력 수신
time.sleep(5)

# STT 스레드로 음성 입력 중지
stt_thread.stop_listening()

# STT 결과 출력
result_text = stt_thread.get_result()
print("인식 결과:", result_text)

# 음성 인식 결과를 파일로 저장하는 코드 (UTF-8 인코딩 사용)
filename = "STT_result.txt"  # 저장할 파일명

with open(filename, "w", encoding="utf-8") as file:
    file.write(result_text)






# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # 다시 'q'를 누르면 재개
#         # 일시정지
#         webcam_thread.pause()
#         break





webcam_thread.resume()



webcam_thread.join()



