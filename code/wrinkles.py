import cv2
import dlib
import numpy as np

# dlib의 얼굴 랜드마크 예측 모델 경로 지정 (.dat 파일 필요)
model_path = r'C:\Users\User\Desktop\python\face_detect\code\shape_predictor_68_face_landmarks.dat'

# 테스트할 얼굴 이미지 경로
image_path = r'C:\Users\User\Desktop\python\face_detect\code\test1.jpg'

# dlib의 얼굴 검출기 로드
detector = dlib.get_frontal_face_detector()

# 68개 얼굴 랜드마크 예측기 로드
predictor = dlib.shape_predictor(model_path)

# 이미지 로드 (BGR 형식)
image = cv2.imread(image_path)

# 그레이스케일 변환 (얼굴 검출 속도와 정확도 향상을 위해)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 영역 검출
faces = detector(gray)

# 검출된 얼굴마다 반복
for face in faces:
    # 얼굴 랜드마크 예측 (68개 점)
    landmarks = predictor(gray, face)

    # 각 랜드마크 위치에 점(초록색) 표시
    for n in range(landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# 주름 측정 예시: 눈 주변 영역 선택
# 왼쪽 눈(36~41), 오른쪽 눈(42~47) 좌표 추출
left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

# 왼쪽 눈 영역의 일부를 관심 영역(ROI)으로 설정
# [y 범위, x 범위] 순으로 지정
# 좌표는 단순히 눈 주변 일부 픽셀만 추출하게 설정
eye_roi = gray[left_eye[1][1]:left_eye[5][1], left_eye[0][0]:left_eye[3][0]]

# 캐니 엣지 검출기로 눈 주변의 주름 및 윤곽선 검출
edges = cv2.Canny(eye_roi, 50, 150)

# 주름 정도를 "엣지(윤곽선) 픽셀의 비율"로 추정
wrinkle_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

# 주름 비율 출력
print(f"주름 비율: {wrinkle_density:.4f} %")
