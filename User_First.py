import openai
import sqlite3
from collections import Counter

# 파일 읽기
with open('emotion_check.txt', 'r') as f:
    lines = f.readlines()

    # 표정과 만족도 추출
    emotions = []
    for i in range(0, len(lines), 1):
        emotion = lines[i].strip().split(': ')[1]
        emotions.append(emotion)

    # 등장 횟수 계산
    emotion_counts = Counter(emotions)

    # 과반수인 표정과 만족도 출력
    majority_emotion = max(emotion_counts, key=emotion_counts.get)

    print("\n과반수인 표정:", majority_emotion)

    # 결과를 파일로 저장
    with open('emotion_result.txt', 'w', encoding="utf-8") as f:
        f.write(f"표정: {majority_emotion}\n")


# Set your OpenAI API key
openai.api_key = "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"

messages = []

# 다른 프로젝트에서 파일을 읽어 결과 사용하는 코드
emotion = "emotion_result.txt"  # 표정 인식 결과가 저장된 파일명
stt = "STT_result.txt" # 음성 인식 결과가 저장된 파일명

with open(emotion, "r", encoding="utf-8") as EMO:
    EMO = EMO.read()

with open(stt, "r", encoding="utf-8") as STT:
    STT = STT.read()

with open("HATE.txt", "r", encoding="utf-8") as file:
    HATE = file.read()

# result 변수에는 파일에서 읽어온 결과가 저장됨
print("\n사용자 표정 결과:", EMO)
print("\n사용자 질문:", STT)


emotion = EMO
question = STT

if 'happy' in emotion:
    emotion = emotion.replace('표정: happy', '행복해')
if 'angry' in emotion:
    emotion = emotion.replace('표정: angry', '화가나')
if 'sad' in emotion:
    emotion = emotion.replace('표정: sad', '우울해')
if 'surprise' in emotion:
    emotion = emotion.replace('표정: surprise', '깜짝 놀라')
if 'neutral' in emotion:
    emotion = emotion.replace('표정: neutral', '중립적이여')


with open("USER.txt", "r", encoding="utf-8") as file:
    USER = file.read()

# 데이터베이스 연결
conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

# 사용자의 mbti 정보 가져오기
cursor.execute("SELECT q1 FROM main_user WHERE password = ?", (USER,))
gender_result = cursor.fetchone()

cursor.execute("SELECT q2 FROM main_user WHERE password = ?", (USER,))
age_result = cursor.fetchone()

cursor.execute("SELECT q3 FROM main_user WHERE password = ?", (USER,))
accent_result = cursor.fetchone()

cursor.execute("SELECT q4 FROM main_user WHERE password = ?", (USER,))
answer_result = cursor.fetchone()

cursor.execute("SELECT q5 FROM main_user WHERE password = ?", (USER,))
mbti_result = cursor.fetchone()

cursor.execute("SELECT q6 FROM main_user WHERE password = ?", (USER,))
hobby_result = cursor.fetchone()

cursor.execute("SELECT q7 FROM main_user WHERE password = ?", (USER,))
music_result = cursor.fetchone()


if gender_result:
    gender = gender_result[0]

if age_result:
    age = age_result[0]

if accent_result:
    accent = accent_result[0]

if answer_result:
    answer = answer_result[0]

if mbti_result:
    mbti = mbti_result[0]

if hobby_result:
    hobby = hobby_result[0]

if music_result:
    music = music_result[0]


content = "이 사람의 성별은 " + gender + "이고 나이는 " + age +"야. 말투는 " + \
          accent + "를 선호하고 답변 형식은 " + answer + "을 선호해. mbti는 " + \
          mbti + "이고 취미는 " + hobby + "이며 좋아하는 음악장르는 " + \
          music + "야. 이사람은" + HATE + "에 관련된 답변은 싫어하고 이 사람의 현재 표정은 "+ \
          emotion + " 보여. 이사람이 " + \
          question + " 라고 물어봤는데 물어본 것에 관한 답변 5개 생성해줘"


# content = "지금 표정이 " + emotion + " 보이고 mbti가 " + mbti + "인  사람이 " + question + " 라고 물어봤는데 " + HATE + " 에 관련된 말을 제외하고 이에관한 답변 5개 생성해줘"
messages.append({"role": "user", "content": content})


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

chat_response = completion.choices[0].message.content
print(f'ChatGPT:\n{chat_response}')
messages.append({"role": "assistant", "content": chat_response})

# 결과를 파일로 저장
with open('chatGPT_result.txt', 'w', encoding="utf-8") as f:
    f.write(f"{chat_response}\n")

from gtts import gTTS
import os
import random

# File path of the GPT result file
filename = "chatGPT_result.txt"

# Read the contents of the file with UTF-8 encoding
with open(filename, "r", encoding="utf-8") as file:
    GPT = file.read()

# Split the text into lines
lines = GPT.splitlines()

# Randomly select a line from lines list
random_line = random.choice(lines)

# 결과를 파일로 저장
with open('TTS.txt', 'w', encoding="utf-8") as f:
    f.write(f"{random_line}\n")

# 1,2,3,4,5X
if '1.' in random_line:
    tts_text = random_line.replace("1.", "")
if '2.' in random_line:
    tts_text = random_line.replace("2.", "")
if '3.' in random_line:
    tts_text = random_line.replace("3.", "")
if '4.' in random_line:
    tts_text = random_line.replace("4.", "")
if '5.' in random_line:
    tts_text = random_line.replace("5.", "")


# # Print the selected line
# print(random_line)

# Create a gTTS object
tts = gTTS(text=tts_text, lang='ko')

# Output file path for the MP3 file
output_file = "output.mp3"

# Save the TTS as an MP3 file
tts.save(output_file)

# Play the MP3 file
os.system(f"start {output_file}")

import time

time.sleep(5)

import subprocess

subprocess.Popen(['python', 'Satisfaction_check.py'])

