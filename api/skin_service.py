# api/skin_service.py
from pathlib import Path
from typing import Union

from PIL import Image

from acne.infer_acne import AcnePredictor

from skin_features.lips import LipDrynessAnalyzer
from skin_features.pores import PoreAnalyzer
from skin_features.wrinkles import WrinkleAnalyzer

PathLike = Union[str, Path]


class SkinAnalysisService:
    def __init__(
        self,
        acne_ckpt_path: str = "acne/acne_resnet50_best.pth",
        device: str | None = None,
    ):
        # 여드름 모델
        self.acne_predictor = AcnePredictor(acne_ckpt_path, device=device)

        # Mediapipe 기반 분석기들
        self.lip_analyzer = LipDrynessAnalyzer()
        self.pore_analyzer = PoreAnalyzer()
        self.wrinkle_analyzer = WrinkleAnalyzer()

    def _to_image(self, x: PathLike | Image.Image) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        return Image.open(x).convert("RGB")

    def analyze_acne(self, front_img, left_img, right_img):

        def _analyze_single(img):
            # 혹시 str/Path가 올 수도 있으니 한 번 통일
            pil_img = self._to_image(img)

            out = self.acne_predictor.predict_pil(pil_img)
            probs = out.get("probs", {})

            p_acne = float(probs.get("acne", 0.0))
            p_pimple = float(probs.get("pimple", 0.0))
            p_spot = float(probs.get("spot", 0.0))

            # API 명세서에 정의된 가중합 방식
            # severity = 1.0 * p_acne + 0.6 * p_pimple + 0.3 * p_spot
            severity = 1.0 * p_acne + 0.6 * p_pimple + 0.3 * p_spot
            # 혹시라도 float 오차로 0~1 살짝 넘는 걸 방지
            severity = max(0.0, min(1.0, severity))

            return {
                "pred_class": out.get("pred_class"),
                "probs": probs,
                "severity": severity,
            }

        # 각 뷰별 결과
        front_res = _analyze_single(front_img)
        left_res = _analyze_single(left_img)
        right_res = _analyze_single(right_img)

        # severity 평균 = overall_severity
        overall_severity = (
            front_res["severity"] + left_res["severity"] + right_res["severity"]
        ) / 3.0

        acne_block = {
            "front": front_res,
            "left": left_res,
            "right": right_res,
            "overall_severity": overall_severity,
        }

        # analyze()에서 self.analyze_acne(... )["acne"] 로 쓰고 있으므로
        return {"acne": acne_block}

    # ------- 추가: 입술/주름/모공 view별 분석 래퍼 -------
    def _analyze_three_views(self, analyzer, front_img, left_img, right_img) -> dict:
        """
        공통 패턴: 분석기(analyzer)가 analyze_image(img) -> score(float|None)를
        반환한다고 가정하고, 3뷰 + overall_severity 블록으로 감싸준다.
        """
        def safe(analyze_fn, img):
            try:
                v = analyze_fn(img)
            except Exception:
                v = None
            return v

        f = safe(analyzer.analyze_image, front_img)
        l = safe(analyzer.analyze_image, left_img)
        r = safe(analyzer.analyze_image, right_img)

        vals = [v for v in (f, l, r) if v is not None]
        overall = float(sum(vals) / len(vals)) if vals else None

        def pack(v):
            return None if v is None else {"severity": float(v)}

        return {
            "front": pack(f),
            "left": pack(l),
            "right": pack(r),
            "overall_severity": overall,
        }

    # ------- 최종 통합 analyze -------
    def analyze(
        self,
        front: PathLike | Image.Image,
        left: PathLike | Image.Image,
        right: PathLike | Image.Image,
    ):
        """
        전체 피부 분석 (여드름 + 주름 + 모공 + 입술 건조도)
        """
        front_img = self._to_image(front)
        left_img = self._to_image(left)
        right_img = self._to_image(right)

        # 1) 여드름
        acne_block = self.analyze_acne(front_img, left_img, right_img)["acne"]

        # 2) 주름 / 모공 / 입술건조도
        wrinkle_block = self._analyze_three_views(
            self.wrinkle_analyzer, front_img, left_img, right_img
        )
        pores_block = self._analyze_three_views(
            self.pore_analyzer, front_img, left_img, right_img
        )
        lip_block = self._analyze_three_views(
            self.lip_analyzer, front_img, left_img, right_img
        )

        return {
            "acne": acne_block,
            "wrinkle": wrinkle_block,
            "pores": pores_block,
            "lip_dryness": lip_block,
        }
