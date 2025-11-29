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

       # 여드름 심각도 점수를 확률의 가중합으로 계산
        def severity_score(res:dict) ->float:
            probs = res["probs"]

            p_acne = float(probs.get("acne", 0.0)) # 여드름
            p_pimple = float(probs.get("pimple", 0.0)) # 뾰루지
            p_spot = float(probs.get("spot", 0.0)) # 반점, 자국

            # 가중치(acne: 1.0(가장 심함) / pimple: 0.6(중간) / spot: 0.3(흔적, 자국))
            score = 1.0 * p_acne + 0.6 * p_pimple + 0.3 * p_spot

            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0

            return score  


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

    # 백엔드에 넘겨줄 분석 결과 가공(우선 나머지 비워두고 acne만)
    def analyze(
        self,
        front: PathLike | Image.Image,
        left: PathLike | Image.Image,
        right: PathLike | Image.Image,
    ):
        """
        전체 피부 분석 (여드름 + 주름 + 모공 + 입술 건조도)

        현재는 여드름만 실제 값이 들어가고,
        wrinkle / pores / lip_dryness 는 나중에 Mediapipe 코드 들어오면 채울 예정.
        """
        acne_block = self.analyze_acne(front, left, right)["acne"]

        # TODO: 나중에 Mediapipe 모듈이 준비되면 아래 None 부분만 교체하면 됨.
        wrinkle_block = None
        pores_block = None
        lip_block = None

        return {
            "acne": acne_block,
            "wrinkle": wrinkle_block,
            "pores": pores_block,
            "lip_dryness": lip_block,
        }