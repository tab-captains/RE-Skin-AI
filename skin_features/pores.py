from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


class PoreAnalyzer:
    """
    얼굴 이미지에서 모공 밀도(pore density)를 계산하는 분석기.

    - 결과: 0.0 ~ 1.0 사이의 severity (1에 가까울수록 모공이 많다고 가정)
    """

    def _to_bgr(self, img: Union[Image.Image, PathLike]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            bgr = cv2.imread(str(img))
            if bgr is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {img}")
            return bgr
        elif isinstance(img, Image.Image):
            rgb = np.array(img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"지원하지 않는 타입: {type(img)}")

    def analyze_image(self, img: Union[Image.Image, PathLike]) -> Optional[float]:
        try:
            bgr = self._to_bgr(img)
        except Exception:
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)

        # CLAHE로 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)

        # 모공 후보 영역 추출 (adaptive threshold)
        binary_pores = cv2.adaptiveThreshold(
            contrast_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 침식으로 작은 노이즈 제거
        kernel_erode = np.ones((2, 2), np.uint8)
        filtered_pores = cv2.erode(binary_pores, kernel_erode, iterations=1)

        # 피부 영역 마스크
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)

        kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_mask, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_mask, iterations=1)

        filtered_pores_final = cv2.bitwise_and(filtered_pores, filtered_pores, mask=skin_mask)

        pore_pixels = np.sum(filtered_pores_final == 255)
        total_pixels = filtered_pores_final.shape[0] * filtered_pores_final.shape[1]
        if total_pixels == 0:
            return None

        pore_density_percent = (pore_pixels / total_pixels) * 100.0  # 0~100%

        # 0~1 스케일로 정규화 (그냥 100으로 나눔)
        severity = pore_density_percent / 100.0
        if severity > 1.0:
            severity = 1.0
        if severity < 0.0:
            severity = 0.0

        return float(severity)
