import openai
import cv2
import numpy as np

# Set your OpenAI API key
openai.api_key = "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"

messages = []

# 다른 프로젝트에서 파일을 읽어 결과 사용하는 코드
filename = "emotion_result.txt"  # 음성 인식 결과가 저장된 파일명

with open(filename, "r", encoding="utf-8") as file:
    result = file.read()

# result 변수에는 파일에서 읽어온 결과가 저장됨
print("사용자 표정 결과:", result)


emotion = result

content = "지금 표정이" + emotion + "한 사람에게 먼저 할 수 있는 말 5개 생성해줘"
messages.append({"role": "user", "content": content})

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

chat_response = completion.choices[0].message.content
print(f'ChatGPT: {chat_response}')
messages.append({"role": "assistant", "content": chat_response})

# 결과를 파일로 저장
with open('chatGPT_result.txt', 'w', encoding="utf-8") as f:
    f.write(f"{chat_response}\n")

