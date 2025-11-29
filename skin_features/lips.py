# skin_features/lips_mp.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


class LipDrynessAnalyzer:
    """
    Mediapipe Face Mesh를 사용해서 입술 주변의 '밝은 픽셀 비율'을 계산.
    - 결과값: 0.0 ~ 1.0 (1에 가까울수록 건조/각질 많다고 해석)
    """

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        # FACEMESH_LIPS에서 입술 인덱스 자동 수집
        lips_connections = mp.solutions.face_mesh.FACEMESH_LIPS
        lip_indices = set()
        for i, j in lips_connections:
            lip_indices.add(i)
            lip_indices.add(j)
        self.lip_indices = sorted(list(lip_indices))

    def _to_rgb(self, img: Union[Image.Image, PathLike]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            bgr = cv2.imread(str(img))
            if bgr is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {img}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        elif isinstance(img, Image.Image):
            return np.array(img.convert("RGB"))
        else:
            raise TypeError(f"지원하지 않는 타입: {type(img)}")

    def analyze_image(self, img: Union[Image.Image, PathLike]):
        """
        한 장의 얼굴 이미지에서 입술 건조도 점수를 계산.
        - 리턴: 0.0 ~ 1.0 (float) / 얼굴 또는 입술을 못 찾으면 None
        """
        rgb = self._to_rgb(img)
        h, w, _ = rgb.shape

        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark

        # 입술 좌표 (pixel)
        points = []
        for idx in self.lip_indices:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))

        if len(points) < 3:
            # 다각형을 만들기 어려운 경우
            return None

        points = np.array(points, dtype=np.int32)

        # 입술 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # 입술 영역만 추출해서 밝기 분석
        lip_region = cv2.bitwise_and(rgb, rgb, mask=mask)
        gray = cv2.cvtColor(lip_region, cv2.COLOR_RGB2GRAY)

        # "밝은 픽셀" (각질/건조 부위로 가정) – 임계값은 경험적으로 190 정도 사용
        _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

        total_pixels = cv2.countNonZero(mask)
        white_pixels = cv2.countNonZero(white_mask)

        if total_pixels == 0:
            return None

        ratio = white_pixels / float(total_pixels)  # 0~1

        # 혹시 모를 이상값 클램핑
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0

        return float(ratio)
