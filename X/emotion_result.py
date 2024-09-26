from collections import Counter

# 파일 읽기
with open('captured_results.txt', 'r') as f:
    lines = f.readlines()

    # 표정과 만족도 추출
    emotions = []
    satisfactions = []
    for i in range(0, len(lines), 2):
        emotion = lines[i].strip().split(': ')[1]
        satisfaction = lines[i+1].strip().split(': ')[1]
        emotions.append(emotion)
        satisfactions.append(satisfaction)

    # 등장 횟수 계산
    emotion_counts = Counter(emotions)
    satisfaction_counts = Counter(satisfactions)

    # 과반수인 표정과 만족도 출력
    majority_emotion = max(emotion_counts, key=emotion_counts.get)
    majority_satisfaction = max(satisfaction_counts, key=satisfaction_counts.get)

    print("과반수인 표정:", majority_emotion)
    print("과반수인 만족도:", majority_satisfaction)

    # 결과를 파일로 저장
    with open('emotion_result.txt', 'w') as f:
            f.write(f"Emotion: {majority_emotion}\n")
            f.write(f"Satisfaction: {majority_satisfaction}\n")


    import subprocess

    # 현재 파일 실행 완료 후 실행할 파일 경로
    next_file_path = './chatGPT.py'


    # 현재 파일 실행 완료 후 cmd 명령 실행
    def run_next_file():
        cmd_command = ['python', next_file_path]
        subprocess.Popen(cmd_command)


    # 현재 파일의 실행이 끝났을 때 호출되는 함수
    def on_file_completed():
        # 실행할 코드들...

        # 다음 파일 실행
        run_next_file()


    # 현재 파일의 실행이 끝났을 때 on_file_completed() 함수 호출
    if __name__ == '__main__':
        on_file_completed()