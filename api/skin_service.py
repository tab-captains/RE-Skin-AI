from pathlib import Path
from typing import Union

from PIL import Image

from acne.infer_acne import AcnePredictor


PathLike = Union[str, Path]


class SkinAnalysisService:
    def __init__(
        self,
        acne_ckpt_path: str = "acne/acne_resnet50_best.pth",
        device: str | None = None,
    ):
        # 여드름 모델 로드
        self.acne_predictor = AcnePredictor(acne_ckpt_path, device=device)

    def _to_image(self, x: PathLike | Image.Image) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        return Image.open(x).convert("RGB")

    def analyze_acne(
        self,
        front: PathLike | Image.Image,
        left: PathLike | Image.Image,
        right: PathLike | Image.Image,
    ) -> dict:
        """정면/좌/우 3장의 여드름 분석 결과 반환"""

        front_img = self._to_image(front)
        left_img = self._to_image(left)
        right_img = self._to_image(right)

        front_res = self.acne_predictor.predict_pil(front_img)
        left_res = self.acne_predictor.predict_pil(left_img)
        right_res = self.acne_predictor.predict_pil(right_img)

        # 예시: 전체 severity score = 세 뷰에서 'acne', 'pimple', 'spot' 확률 평균
        def severity_score(res: dict) -> float:
            probs = res["probs"]
            keys = [k for k in probs.keys() if k in ("acne", "pimple", "spot")]
            if not keys:
                return 0.0
            return float(sum(probs[k] for k in keys) / len(keys))

        front_score = severity_score(front_res)
        left_score = severity_score(left_res)
        right_score = severity_score(right_res)

        overall_score = float((front_score + left_score + right_score) / 3.0)

        return {
            "acne": {
                "front": {**front_res, "severity": front_score},
                "left": {**left_res, "severity": left_score},
                "right": {**right_res, "severity": right_score},
                "overall_severity": overall_score,
            }
        }
