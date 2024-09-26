import speech_recognition as sr
from playsound import playsound
import sqlite3
from gtts import gTTS
import os
import time

tts = gTTS(text="안녕이라고 말씀 하신 뒤 전화번호를 말씀해 주세요", lang='ko')

# Output file path for the MP3 file
output_file = "output.mp3"

# Save the TTS as an MP3 file
tts.save(output_file)

# Play the MP3 file
os.system(f"start {output_file}")

time.sleep(5)

# 음성 인식 객체 생성
r = sr.Recognizer()

# 마이크로부터 오디오 입력 받기
with sr.Microphone() as source:
    print("음성 인식 대기 중...")

    while True:
        audio = r.listen(source)

        try:
            # 음성을 텍스트로 변환
            text = r.recognize_google(audio, language='ko-KR')

            # '다울아'를 감지하면 음성 입력 대기
            if '안녕' in text:
                playsound('ding.wav')  # 띠링 소리 재생
                print("다음 음성을 기다리는 중...")
                audio = r.listen(source, phrase_time_limit=5)
                text = r.recognize_google(audio, language='ko-KR')
                break

        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
        except sr.RequestError:
            print("Google 음성 인식 서비스에 접근할 수 없습니다.")


    # '다울아'를 제외한 텍스트 결과 출력
    result_text = text.replace("안녕", "")
    result_text = text.replace(" ", "")
    print(result_text)

    # 음성 인식 결과를 파일로 저장하는 코드 (UTF-8 인코딩 사용)
    result = result_text  # 음성 인식 결과
    filename = "USER.txt"  # 저장할 파일명

    with open(filename, "w", encoding="utf-8") as file:
        file.write(result)

    with open(filename, "r", encoding="utf-8") as file:
        USER = file.read()

    # 데이터베이스 연결
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()

    # 사용자의 mbti 정보 가져오기
    cursor.execute("SELECT Name FROM main_user WHERE password = ?", (USER,))
    Name = cursor.fetchone()

    if Name:
        Name = Name[0]

    TTS = "안녕하세요. " + Name + "님 만나서 반갑습니다.\n 이제 " + Name +"님의 성향을 반영하여 다울이를 실행시키겠습니다."

    tts = gTTS(text= TTS , lang='ko')

    # Output file path for the MP3 file
    output_file = "output.mp3"

    # Save the TTS as an MP3 file
    tts.save(output_file)

    # Play the MP3 file
    os.system(f"start {output_file}")

    time.sleep(5)

    import subprocess

    subprocess.Popen(['python', 'Emotion_check.py'])