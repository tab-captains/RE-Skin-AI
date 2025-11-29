import cv2
import dlib
import numpy as np


# 모델 및 이미지 경로 설정
model_path = r'C:\Users\User\Desktop\python\face_detect\code\shape_predictor_68_face_landmarks.dat'
image_path = r'C:\Users\User\Desktop\python\face_detect\code\test1.jpg'


# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
    exit()


# BGR -> 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# dlib 얼굴 탐지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(model_path)
except RuntimeError as e:
    print(f"모델 파일을 로드할 수 없습니다: {e}")
    exit()


# 입술 마스크 생성 함수
def get_lip_mask(image, shape):
    # 입술 랜드마크(48~67) 추출
    lip_landmarks = shape[48:68]
    # 랜드마크 좌표 리스트 생성
    points = np.array([(p.x, p.y) for p in lip_landmarks], np.int32)
    # 마스크 생성 (초기값: 0)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # 입술 영역 채우기
    cv2.fillPoly(mask, [points], 255)
    return mask


# 입술 각질 비율 계산 함수
def calculate_white_ratio(image, mask):
    # 마스크 적용하여 입술 영역 추출
    lip_region = cv2.bitwise_and(image, image, mask=mask)
    # 그레이스케일 변환
    gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
    # 흰색 픽셀 추출 (임계값 180)
    white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    # 전체 픽셀 수 및 흰색 픽셀 수 계산
    total_pixels = cv2.countNonZero(mask)
    white_pixels = cv2.countNonZero(white_mask)
    # 비율 계산
    ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    return ratio


# 얼굴 탐지
faces = detector(gray)
if len(faces) == 0:
    print("이미지에서 얼굴을 찾을 수 없습니다.")
else:
    for face in faces:
        # 랜드마크 추출
        shape = predictor(gray, face)
        # 입술 마스크 생성
        lip_mask = get_lip_mask(image, shape.parts())
        # 입술 각질 비율 계산
        white_ratio = calculate_white_ratio(image, lip_mask)
        print(f"입술 각질 비율: {white_ratio:.4f} %")
        # 이미지 창 표시 부분 삭제