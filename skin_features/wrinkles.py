# skin_features/wrinkles.py
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

PathLike = Union[str, Path, Image.Image]


class WrinkleAnalyzer:
    """
    Mediapipe Face Mesh를 사용해서 왼쪽 눈/눈썹 주변 ROI의 edge 비율을 계산.
    - 결과값: 0.0 ~ 1.0 (1에 가까울수록 주름이 많은 것으로 해석)
    """

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # 왼쪽 눈 + 왼쪽 눈썹 인덱스 모으기
        self.left_indices = self._collect_left_eye_and_eyebrow_indices()

    @staticmethod
    def _collect_left_eye_and_eyebrow_indices() -> set[int]:
        mesh = mp.solutions.face_mesh
        indices: set[int] = set()

        # FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW의 연결 정보에서 인덱스만 추출
        for conn in (mesh.FACEMESH_LEFT_EYE, mesh.FACEMESH_LEFT_EYEBROW):
            for i, j in conn:
                indices.add(i)
                indices.add(j)

        return indices

    @staticmethod
    def _to_bgr(img: PathLike) -> np.ndarray:
        """입력을 BGR numpy array로 통일."""
        if isinstance(img, (str, Path)):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def analyze_image(self, img: PathLike) -> Optional[float]:
        """
        한 장의 이미지에 대해 주름 정도를 0.0~1.0 사이 float로 반환.
        얼굴/랜드마크를 못 찾으면 None.
        """
        bgr = self._to_bgr(img)
        h, w, _ = bgr.shape

        # Mediapipe는 RGB 입력을 기대함
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # 얼굴을 못 찾은 경우
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 왼쪽 눈/눈썹 인덱스를 기준으로 bounding box 계산
        xs, ys = [], []
        for idx in self.left_indices:
            lm = landmarks[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        if not xs or not ys:
            return None

        x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

        roi = bgr[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge
        edges = cv2.Canny(gray, 50, 150)

        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        if total_pixels == 0:
            return None

        edge_ratio = edge_pixels / total_pixels  # 보통 0.x 대 값

        # 간단히 스케일 조정 후 0~1로 클램프 (나중에 튜닝 가능)
        severity = float(np.clip(edge_ratio * 5.0, 0.0, 1.0))
        return severity
