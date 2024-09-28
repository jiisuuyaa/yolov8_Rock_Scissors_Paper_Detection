import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="YOLO Model Demo", layout="centered")

# 페이지 제목 및 설명
st.title("🤸‍♀️YOLOv8")
st.title("Rock✊, Scissors✌, Paper🖐 Detection")
st.write("""
YOLOv8 모델을 사용하여 가위, 바위, 보를 실시간으로 분류합니다.
이 모델은 Roboflow를 통해 학습된 데이터셋을 사용하며, 다음과 같은 과정을 거쳤습니다:
- 데이터 수집 및 라벨링
- 모델 학습 및 검증
- 웹캠을 통한 실시간 객체 탐지
""")

# 수평으로 이미지 정렬
col1, col2 = st.columns(2)

# 각 컬럼에 이미지 추가
with col1:
    st.image("https://github.com/user-attachments/assets/cbc639f2-7055-419e-a94e-6e1fbe143b61", 
             caption="Rock, Paper, Scissors Dataset", use_column_width=True)

with col2:
    st.image("https://github.com/user-attachments/assets/3840f1b6-06ce-401b-b242-1dc0e1dbf891", 
             caption="YOLOv8 Model", use_column_width=True)

# 모델 설명 추가
st.write("""
위의 이미지는 학습된 데이터셋과 YOLOv8 모델의 구조를 나타냅니다.
실시간 웹캠에서 가위✌, 바위✊, 보🖐 중 하나를 내면
YOLOv8 모델을 통해 분류할 수 있습니다 !✨
""")

# 웹캠 선택 및 설정
st.header("웹캠을 통해 실시간 분류하기")
camera_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2])

# 모델 파일 경로 설정
model = YOLO('./custom_train/yolov8n_rock_paper_scissors.pt')  # 모델 파일 경로

# 모델 클래스 이름 출력
st.sidebar.text("Model Classes:")
st.sidebar.write(model.names)

# 웹캠 스트리밍 시작 및 중지 상태를 관리하는 변수
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# 웹캠 스트리밍 상태에 따른 처리
if st.session_state.streaming:
    cap = cv2.VideoCapture(camera_index)
    stframe = st.empty()  # Streamlit에서 사용할 빈 이미지 프레임 설정

    while True:
        success, frame = cap.read()
        if not success:
            st.error("웹캠에서 프레임을 읽어올 수 없습니다.")
            break

        # 객체 탐지 (Rock, Paper, Scissors 클래스 탐지)
        results = model.predict(frame, classes=[0, 1, 2], conf=0.4, imgsz=640)

        # 탐지된 결과 시각화
        annotated_frame = results[0].plot()

        # BGR 이미지를 RGB로 변환
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Streamlit을 통해 이미지 표시
        stframe.image(annotated_frame, channels="RGB")

        # 종료 조건 설정 (웹캠 스트리밍 중지)
        if not st.session_state.streaming:
            break

    cap.release()  # 웹캠 해제
    st.write("Webcam streaming stopped.")

# 웹캠 스트리밍 시작 버튼
if st.button("Start Webcam"):
    st.session_state.streaming = True

# 웹캠 스트리밍 중지 버튼
if st.button("Stop Webcam"):
    st.session_state.streaming = False
