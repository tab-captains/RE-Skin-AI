import cv2
import numpy as np


def analyze_pores(image_path):
    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print(f"오류: '{image_path}' 파일을 찾을 수 없거나 로드할 수 없습니다.")
            return

        # 컬러 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        return

    # 이미지 전처리 (노이즈 제거 및 대비 향상)
    denoised = cv2.medianBlur(gray, 3)

    # CLAHE(대비 평탄화)를 사용하여 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # 모공 추출
    binary_pores = cv2.adaptiveThreshold(
        contrast_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 형태학적 연산 (침식, Erode)
    kernel_erode = np.ones((2, 2), np.uint8)
    filtered_pores = cv2.erode(binary_pores, kernel_erode, iterations=1)

    # YCrCb 색 공간으로 변환
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 피부색 범위 설정
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)

    # 피부 영역 마스크 생성
    skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)

    # 마스크 클리닝
    kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_mask, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_mask, iterations=1)

    # 모공 이미지에 Skin Mask 적용
    filtered_pores_final = cv2.bitwise_and(filtered_pores, filtered_pores, mask=skin_mask)

    # 결과 정량화
    pore_pixels = np.sum(filtered_pores_final == 255)
    total_pixels = filtered_pores_final.shape[0] * filtered_pores_final.shape[1]
    pore_density = (pore_pixels / total_pixels) * 100  # %로 표시

    # 모공 비율 출력
    print(f"모공 비율: {pore_density:.4f} %")


if __name__ == "__main__":
    image_file = r'C:\Users\User\Desktop\python\face_detect\code\test1.jpg'
    analyze_pores(image_file)
