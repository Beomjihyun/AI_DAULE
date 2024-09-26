from gtts import gTTS
import os
import random

#파일 불러오기----------------------------------------------------

# File path of the GPT result file
filename = "chatGPT_result.txt"

# Read the contents of the file with UTF-8 encoding
with open(filename, "r", encoding="utf-8") as file:
    GPT = file.read()

#---------------------------------------------------------------

# 랜덤으로 결과 선택------------------------------------------------
# Split the text into lines
lines = GPT.splitlines()

# Randomly select a line from lines list
random_line = random.choice(lines)

# Print the selected line
print(random_line)

#---------------------------------------------------------------------

# 랜덤결과 TTS----------------------------------------------------------

# Create a gTTS object
tts = gTTS(text=random_line, lang='ko')

# Output file path for the MP3 file
output_file = "output.mp3"

# Save the TTS as an MP3 file
tts.save(output_file)

# Play the MP3 file
os.system(f"start {output_file}")

#---------------------------------------------------------------------

# from gtts import gTTS
# import os
#
# # 다른 프로젝트에서 파일을 읽어 결과 사용하는 코드
# filename = "chatGPT_result.txt"  # GPT 결과가 저장된 파일명
#
# with open(filename, "r") as file:
#     GPT = file.read()
#
# # result 변수에는 파일에서 읽어온 결과가 저장됨
# print(GPT)
#
# # TTS를 생성할 텍스트
# text = GPT
#
# # gTTS 객체 생성
# tts = gTTS(text=text, lang='ko')
#
# # MP3 파일로 변환
# output_file = "output.mp3"
# tts.save(output_file)
#
# # MP3 파일 재생
# os.system(f"start {output_file}")
