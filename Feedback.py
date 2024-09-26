from gtts import gTTS
import os
import random

with open('chatGPT_result.txt', 'r', encoding="utf-8") as file:
    GPT_lines = file.readlines()

# 제외할 텍스트를 정의합니다
with open('TTS.txt', 'r', encoding="utf-8") as file:
    excluded_text = file.read()

# 이전에 누적된 텍스트를 파일에서 읽어옵니다
with open('HATE.txt', 'r', encoding="utf-8") as file:
    accumulated_text = file.read()

if excluded_text not in accumulated_text:
    accumulated_text += excluded_text

# 누적된 텍스트를 파일에 작성합니다
with open('HATE.txt', 'w', encoding="utf-8") as file:
    file.write(accumulated_text)

# 특정 텍스트를 제외한 새로운 리스트를 생성합니다
filtered_lines = [line for line in GPT_lines if excluded_text not in line]



# 새로운 파일에 결과를 작성합니다
with open('chatGPT_result.txt', 'w', encoding="utf-8") as file:
    file.writelines(filtered_lines)


# Read the contents of the file with UTF-8 encoding
with open('chatGPT_result.txt', "r", encoding="utf-8") as file:
    TTS = file.read()

print(TTS)

# Split the text into lines
lines = TTS.splitlines()

# Randomly select a line from lines list
random_line = random.choice(lines)

print(random_line)

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
